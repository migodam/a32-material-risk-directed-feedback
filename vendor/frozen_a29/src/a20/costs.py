"""Inclusive job accounting, exclusive records, failed actions charged."""
from __future__ import annotations
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
import json
import os
import time


class BudgetExceeded(RuntimeError):
    pass


def plain(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, float) and not __import__('math').isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


class CostBook:
    """CPU process seconds include host compute in CUDA jobs and BLAS threads.

    GPU occupancy is the complete CUDA job wall time, including host work.
    These two resource limits are separate. Deployment wall never sums nested
    spans. Both limits reserve room for graceful shutdown and publication.
    """
    def __init__(self, path=None, *, device='cpu', prior_cpu=0., prior_gpu=0.,
                 cpu_limit=7200., gpu_limit=43200., cpu_reserve=300.,
                 gpu_reserve=120., metadata=None, enforce=True):
        self.path = None if path is None else Path(path)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.prior_cpu = float(prior_cpu)
        self.prior_gpu = float(prior_gpu)
        self.cpu_limit = float(cpu_limit)
        self.gpu_limit = float(gpu_limit)
        self.cpu_reserve = float(cpu_reserve)
        self.gpu_reserve = float(gpu_reserve)
        self.started_wall = time.perf_counter()
        self.started_cpu = time.process_time()
        self.metadata = metadata or {}
        self.enforce = enforce
        self.counts = Counter()
        self.walls = Counter()
        self.events = 0
        self.stack = []
        self.peak_cpu_rss = 0
        self.peak_gpu_allocated = 0
        self.peak_gpu_reserved = 0
        self.phase = 'online'

    def synchronize(self):
        if self.device == 'cuda':
            import torch
            torch.cuda.synchronize()

    def check(self):
        if self.enforce:
            if self.prior_cpu + time.process_time() - self.started_cpu >= self.cpu_limit - self.cpu_reserve:
                raise BudgetExceeded('CPU_LIMIT_WITH_EPILOGUE_RESERVE')
            if self.device == 'cuda' and self.prior_gpu + time.perf_counter() - self.started_wall >= self.gpu_limit - self.gpu_reserve:
                raise BudgetExceeded('GPU_OCCUPATION_LIMIT_WITH_STOP_RESERVE')
        try:
            import psutil
            m = psutil.Process().memory_info()
            self.peak_cpu_rss = max(self.peak_cpu_rss, int(getattr(m, 'peak_wset', m.rss)), m.rss)
        except ImportError:
            try:
                import resource
                import sys
                raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                self.peak_cpu_rss = max(self.peak_cpu_rss, raw if sys.platform=='darwin' else raw*1024)
            except ImportError:
                pass
        if self.device == 'cuda':
            import torch
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            self.peak_gpu_allocated = max(self.peak_gpu_allocated, allocated)
            self.peak_gpu_reserved = max(self.peak_gpu_reserved, reserved)
            if torch.cuda.memory_allocated() > .9 * torch.cuda.get_device_properties(0).total_memory:
                raise BudgetExceeded('GPU_MEMORY_STOP_90_PERCENT')

    def append(self, row):
        if self.path:
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(plain(row), allow_nan=False) + '\n')
        self.events += 1

    @contextmanager
    def scope(self, phase):
        old = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = old

    @contextmanager
    def span(self, label, **counters):
        self.check()
        self.synchronize()
        start = time.perf_counter()
        cpu = time.process_time()
        frame = {'child_wall': 0.}
        parent = self.stack[-1] if self.stack else None
        self.stack.append(frame)
        self.counts.update(counters)
        row = {'event': label, 'phase': self.phase, 'counters': counters,
               'status': 'FAILED', **self.metadata}
        try:
            yield row
            row['status'] = 'OK'
        except BaseException as exc:
            row['error'] = type(exc).__name__ + ': ' + str(exc)
            raise
        finally:
            self.synchronize()
            wall = time.perf_counter() - start
            row.update(wall_seconds=wall, exclusive_wall_seconds=max(0., wall-frame['child_wall']),
                       process_cpu_seconds=time.process_time()-cpu)
            self.stack.pop()
            if parent:
                parent['child_wall'] += wall
            self.walls[label] += row['exclusive_wall_seconds']
            self.append(row)

    def snapshot(self):
        self.check()
        return {'counts': dict(self.counts), 'exclusive_walls': dict(self.walls)}

    def delta(self, old):
        return {'counts': {k: int(v-old['counts'].get(k, 0)) for k, v in self.counts.items()},
                'exclusive_walls': {k: float(v-old['exclusive_walls'].get(k, 0.)) for k, v in self.walls.items()}}

    def receipt(self):
        return {'wall_seconds': time.perf_counter()-self.started_wall,
                'process_cpu_seconds': time.process_time()-self.started_cpu,
                'gpu_occupation_seconds': time.perf_counter()-self.started_wall if self.device=='cuda' else 0.,
                'counts': dict(self.counts), 'exclusive_walls': dict(self.walls),
                'events': self.events, 'peak_cpu_rss_bytes': self.peak_cpu_rss,
                'CPU_memory_measurement_available': self.peak_cpu_rss>0,
                'peak_gpu_allocated_bytes': self.peak_gpu_allocated,
                'peak_gpu_reserved_bytes': self.peak_gpu_reserved,
                'device': self.device, 'CPU_in_GPU_job_included': True,
                'deployment_wall_definition': 'inclusive parent elapsed; never sum nested spans'}


def budget_history(root):
    root = Path(root)
    rows = []
    for path in sorted((root/'results/jobs').glob('*/job_receipt.json')):
        rows.append(json.loads(path.read_text(encoding='utf-8')))
    return sum(r.get('process_cpu_seconds', 0.) for r in rows), sum(r.get('gpu_occupation_seconds', 0.) for r in rows)


@contextmanager
def job_lock(root):
    path = Path(root)/'runs/gpu.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise BudgetExceeded('EXISTING_PHYSICS_JOB_LOCK; inspect owner, never silently delete') from exc
    with os.fdopen(fd, 'w') as handle:
        json.dump({'pid': os.getpid(), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, handle)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
