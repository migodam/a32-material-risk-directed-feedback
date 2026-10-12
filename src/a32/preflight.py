"""Cache-only source/packing checks; no native solve and no dev targets."""
from __future__ import annotations
from pathlib import Path
import json, platform, sys
import numpy as np
import scipy
from .budget import Book
from . import data

def pack_measurement(value):
    a=np.asarray(value,np.complex128)
    return np.concatenate((a.T.real,a.T.imag),axis=1).reshape(-1)

def material_real(value):
    a=np.asarray(value,np.complex128).reshape(-1)
    return np.r_[a.real,a.imag]

def relative(residual,scale):
    den=float(np.linalg.norm(scale)); num=float(np.linalg.norm(residual))
    return num/den if den else (0. if num==0 else float('inf'))

def run(root):
    root=Path(root); book=Book(root,'preflight'); config=book.config
    with book.span('read_authorized_inputs',saved_cases=32):
        cases=data.online_cases(config)
        train,train_meta=data.train_deltas(config)
    audits=[]
    for case in cases:
        book.check(); bg=case.background
        # A32 data exposes a named known-background mapping, never object truth.
        field=bg['eb']; current=bg['chi_b'][:,None]*field
        L=np.eye(576,dtype=complex)-bg['chi_b'][:,None]*bg['G']
        with book.span('cached_background_identity',cached_background_source_columns=8,cached_receiver_rows=32):
            e={'field':relative(bg['incident']+bg['G']@current-field,field),
               'data':relative(bg['S']@current-bg['yb'],bg['yb']),
               'transfer':relative(bg['Db']@L-bg['S'],bg['S'])}
            complex_J=np.einsum('rn,ns->srn',bg['Db'],field)/np.sqrt(1/576)
            Jr=np.concatenate((complex_J.real,complex_J.imag),axis=1).reshape(512,576)
            Ji=np.concatenate((-complex_J.imag,complex_J.real),axis=1).reshape(512,576)
            e['real_material_J']=relative(np.c_[Jr,Ji]-bg['J'],bg['J'])
        with book.span('paid_F_N3_cache_read',saved_anchor_reads=1):
            cached=data.cached_n3(case)
        e['cached_material']=relative(cached.chi-case.n3,case.n3)
        if any(not np.isfinite(x) or x>1e-10 for x in e.values()):
            book.failure('PREFLIGHT_IDENTITY',e,case_id=case.case_id)
            raise ValueError('PREFLIGHT_NUMERICAL_MISMATCH:'+case.case_id)
        if not case.sigma>0:raise ValueError('REGISTERED20DB_POSITIVE_SIGMA_REQUIRED')
        audits.append({'case_id':case.case_id,'checks':e,'sigma':float(case.sigma),
                       'noise_whitening':float(np.sqrt(2)/case.sigma)})
    report={'status':'PASS','cases':len(cases),'independent_parents':len({x.parent_id for x in cases}),
            'TRAIN':train_meta,'packing':'source-major [Re32,Im32]; material [Re576,Im576]',
            'material_mass':1/576,'dtype':'complex128/float64','background_J_shape':[512,1152],
            'audits':audits,'new_full_wave_states':0,'new_LU':0,'new_LU_RHS':0,
            'native_solver':'unchanged A24/A2 scalar 2D TM; no backend solve at preflight',
            'missing_attachment_files':['CORE_THEOREM.md','a31_r1_checks.py'],
            'source_field_and_power_status':'NATIVE_E1_NOT_RUN_AT_PREFLIGHT',
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                           'platform':platform.platform(),'executable':sys.executable}}
    (root/'results/a32/PREFLIGHT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:report[k] for k in ['status','cases','independent_parents','new_full_wave_states','missing_attachment_files']}))
    return report
