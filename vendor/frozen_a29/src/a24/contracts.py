"""Shared runtime contracts, durable action reservation and wall budgets."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import time
import numpy as np
from a20.costs import CostBook, BudgetExceeded, write_json, plain

class ForbiddenAccess(RuntimeError):
    pass

class NumericalFailure(RuntimeError):
    pass

FORBIDDEN = frozenset({'truth', 'teacher', 'truth_currents', 'full_J', 'full_H',
                      'reference_step', 'support_mask', 'oracle_currents', 'parent_masks'})

RUNTIME_CONFIG_KEYS = frozenset({'master_seed','dtype','device','grid','side','wavenumber',
    'sources','receivers','source_radius','receiver_radius','real_lower','real_upper',
    'imag_lower','imag_upper','feasibility_tolerance','pole_relative_floor',
    'identity_tolerance','backward_residual_tolerance','core_relative_sigma_min',
    'core_condition_cap','orthogonal_rank_rtol','receiver_retained_rank','observable_rank',
    'target_current_rank','degree','primary_return_rank','oversampling','power_iterations',
    'rom_material_probes','rom_probe_smoothing_cells',
    'blocks','max_half_step_retry','prox_tolerance','prox_max_iter','prox_check_every',
    'prox_raise_on_failure','prox_material_scale'})

def online_config(config):
    return {k:v for k,v in config.items() if k in RUNTIME_CONFIG_KEYS}

def deny_oracle_path(path):
    path=Path(path)
    components={part.lower() for part in path.parts}
    if components & {'offline','offline_eval','oracle','teacher','labels'} or any(
            label in path.name.lower() for label in ('oracle','teacher','full_gn')):
        raise ForbiddenAccess('OFFLINE_PATH_IN_RUNTIME')
    return path

def load_observation(path):
    allowed={'data','sigma','chi_b','case_id','noise_seed'}
    path=deny_oracle_path(path)
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)-allowed:
            raise ForbiddenAccess('OFFLINE_OR_UNREGISTERED_NPZ_KEYS')
        return OnlineObservation(f['data'].copy(),float(f['sigma']),str(f['case_id']))

_ACTIVE_LEDGER = None

def get_ledger():
    if _ACTIVE_LEDGER is None:
        raise RuntimeError('NO_ACTIVE_CHARGED_LEDGER')
    return _ACTIVE_LEDGER

@contextmanager
def active_ledger(book):
    global _ACTIVE_LEDGER
    old = _ACTIVE_LEDGER
    _ACTIVE_LEDGER = book
    try:
        yield book
    finally:
        _ACTIVE_LEDGER = old

@dataclass(frozen=True)
class OnlineObservation:
    data: np.ndarray
    sigma: float
    case_id: str
    def __getattr__(self, key):
        if key in FORBIDDEN:
            raise ForbiddenAccess('OFFLINE_FIELD_IN_RUNTIME:' + key)
        raise AttributeError(key)

class Ledger(CostBook):
    def __init__(self, path=None, *, wall_limit=120., prior_wall=0., metadata=None):
        super().__init__(path, device='cpu', enforce=False, metadata=metadata)
        self.wall_limit = float(wall_limit)
        self.prior_wall = float(prior_wall)
        self.deadline = self.started_wall + max(0., self.wall_limit-self.prior_wall)
        self.reservations = {}
    def check(self):
        if time.perf_counter() >= self.deadline:
            raise BudgetExceeded('STAGE_CPU_WALL_DEADLINE')
        super().check()
    def reserve(self, category, limit):
        self.check()
        used = self.reservations.get(category, 0)
        if used >= int(limit):
            raise BudgetExceeded('ACTION_CAP:' + category)
        self.reservations[category] = used+1
        self.append({'event':'RESERVED', 'category':category, 'used':used+1,
                     'limit':int(limit), 'status':'RESERVED', **self.metadata})
    def receipt(self):
        r=super().receipt()
        r.update(stage_wall_limit_seconds=self.wall_limit,
                 prior_stage_wall_seconds=self.prior_wall,
                 durable_reservations=dict(self.reservations), gpu_seconds=0.)
        return r

def pack(data):
    """(q,s) complex -> per-source [Re q, Im q] real."""
    d=np.asarray(data)
    if d.ndim != 2 or not np.iscomplexobj(d):
        raise ValueError('COMPLEX_Q_BY_S_REQUIRED')
    return np.concatenate((d.T.real,d.T.imag),axis=1).reshape(-1)

def unpack(data, q, s):
    d=np.asarray(data)
    if np.iscomplexobj(d) or d.shape != (2*q*s,):
        raise ValueError('REAL_PACKED_DATA_REQUIRED')
    d=d.reshape(s,2*q)
    return (d[:,:q]+1j*d[:,q:]).T

def mass_pack(chi, volume):
    c=np.asarray(chi)
    return np.sqrt(volume)*np.r_[c.real,c.imag]

def mass_unpack(x, volume):
    x=np.asarray(x); n=len(x)//2
    if np.iscomplexobj(x) or x.ndim!=1 or len(x)!=2*n:
        raise ValueError('REAL_MATERIAL_MASS_COORDINATES_REQUIRED')
    return (x[:n]+1j*x[n:])/np.sqrt(volume)

def relative_error(a,b):
    den=float(np.linalg.norm(b))
    return float(np.linalg.norm(a-b)/den) if den>0 else None

def guard_core(matrix, config, name='core'):
    from scipy.linalg import svdvals
    singular=svdvals(np.asarray(matrix))
    largest=float(singular[0]); smallest=float(singular[-1])
    ratio=smallest/largest if largest>0 else 0.
    condition=largest/smallest if smallest>0 else float('inf')
    record={'name':name,'sigma_min':smallest,'sigma_max':largest,
            'relative_sigma_min':ratio,'condition':condition}
    if (not np.isfinite(condition) or ratio<config.get('core_relative_sigma_min',1e-10)
            or condition>config.get('core_condition_cap',1e10)):
        raise NumericalFailure('UNSAFE_CORE:' + name + ':' + str(record))
    return record
