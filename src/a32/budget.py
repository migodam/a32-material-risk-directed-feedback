"""A32 independent accounting; every native call is reserved before execution."""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import json, os, time

class BudgetStop(RuntimeError):
    pass

def read_rows(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []

def charged_cpu(root):
    return sum(float(r.get('charged_seconds', 0)) for r in read_rows(Path(root)/'results/a32/COST_LEDGER.jsonl'))

def append_jsonl(path, row):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); f.flush()

class Book:
    def __init__(self, root, stage):
        self.root=Path(root).resolve(); self.stage=str(stage)
        self.config=json.loads((self.root/'a32_config.json').read_text())
        self.counts=Counter(); self.start=time.perf_counter(); self.cpu_start=time.process_time()
        self.initial_cpu=charged_cpu(root)
        self.out=self.root/'results/a32'; self.out.mkdir(parents=True,exist_ok=True)

    def check(self):
        current=max(time.perf_counter()-self.start,time.process_time()-self.cpu_start)
        recorded=max(self.initial_cpu,charged_cpu(self.root))
        if recorded+current>=self.config['budgets']['cpu_charged_seconds']:
            raise BudgetStop('CPU_1800_SECONDS_EXHAUSTED')

    def _reserve(self,event,amount,**metadata):
        self.check()
        if isinstance(amount,bool) or not isinstance(amount,int) or amount<1:
            raise ValueError('POSITIVE_INTEGER_RESERVATION_REQUIRED')
        if self.stage not in {'shadow','trajectories','native-check','vector-check'}:
            raise BudgetStop('NATIVE_PHYSICS_FORBIDDEN_IN_'+self.stage.upper())
        gate=self.out/'E0_GATE.json'
        if not gate.is_file() or json.loads(gate.read_text()).get('status')!='PASS':
            raise BudgetStop('E0_PASS_REQUIRED')
        path=self.out/'PHYSICAL_STATE_LEDGER.jsonl'; lock=path.with_suffix('.lock')
        try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError: raise BudgetStop('ACCOUNTING_LOCK_BUSY') from None
        try:
            os.write(fd,str(os.getpid()).encode()); os.close(fd)
            rows=read_rows(path); budgets=self.config['budgets']
            used=sum(r['amount'] for r in rows if r['event']==event)
            limit=budgets['material_states_total'] if event=='STATE_RESERVED' else budgets['complex_RHS']
            if used+amount>limit: raise BudgetStop('UNIFIED_'+event+'_CAP')
            if event=='STATE_RESERVED':
                dim=metadata['physical_dimension']
                du=sum(r['amount'] for r in rows if r['event']==event and r.get('physical_dimension')==dim)
                if du+amount>budgets['states_2d' if dim==2 else 'states_3d']:
                    raise BudgetStop('DIMENSION_STATE_CAP')
                stage_cap={'shadow':24,'trajectories':28,'native-check':4,'vector-check':8}[self.stage]
                su=sum(r['amount'] for r in rows if r['event']==event and r.get('stage')==self.stage)
                if su+amount>stage_cap: raise BudgetStop('STAGE_STATE_CAP')
            row=dict(event=event,amount=amount,cumulative=used+amount,limit=limit,stage=self.stage,
                     timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**metadata)
            append_jsonl(path,row); self.counts[event]+=amount
            return row
        finally: lock.unlink(missing_ok=True)

    def reserve_state(self, material=None, *, resolution=24, purpose='unspecified',count=1,physical_dimension=2):
        return self._reserve('STATE_RESERVED',int(count),physical_dimension=int(physical_dimension),
                             resolution=int(resolution),purpose=str(purpose),material_shape=list(getattr(material,'shape',())))

    def reserve_rhs(self,count,role='unspecified'):
        return self._reserve('RHS_RESERVED',int(count),role=str(role))

    @contextmanager
    def span(self,label,**counts):
        self.check(); start=time.perf_counter(); cpu=time.process_time(); status='FAILED'
        self.counts.update(counts)
        try: yield; status='COMPLETE'
        finally:
            append_jsonl(self.out/'ACTIONS.jsonl',dict(event='ACTION',label=label,status=status,stage=self.stage,
                        counts=counts,wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu))

    def failure(self,label,error,**extra):
        append_jsonl(self.out/'FAILURE_LEDGER.jsonl',dict(stage=self.stage,label=label,error=str(error),**extra))

def totals(root):
    rows=read_rows(Path(root)/'results/a32/PHYSICAL_STATE_LEDGER.jsonl')
    return {'new_material_state_evaluations':sum(r['amount'] for r in rows if r['event']=='STATE_RESERVED'),
            'new_LU_RHS':sum(r['amount'] for r in rows if r['event']=='RHS_RESERVED'),
            'charged_cpu_seconds':charged_cpu(root),'gpu_occupied_seconds':sum(float(r.get('gpu_occupied_seconds',0)) for r in read_rows(Path(root)/'results/a32/COST_LEDGER.jsonl'))}
