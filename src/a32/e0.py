"""Frozen continuous L/anchor-L material fitting, then independent truth scoring."""
from __future__ import annotations
from pathlib import Path
import csv,json,time
import numpy as np
from scipy import linalg as la
from .budget import Book,append_jsonl
from .preflight import pack_measurement,material_real
from . import data,charts
from .priors import build_priors
from .qp import solve_qp

def clean(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def _c(result):
    return np.asarray(result['c'],float)

def gate_from_statistics(result, cfg):
    """One authoritative coverage guard; ancillary metrics are not new gates."""
    checked=result['E0_gate']
    complete=bool(checked['coverage_complete'])
    status=checked['status'] if complete else 'HOLD'
    primary=[r for r in result['parent_rows'] if r['method']==cfg['primary_method']
             and r['family'] in cfg['primary_families']]
    return {'status':status,'primary_method':cfg['primary_method'],
            'relative_equal_parent_MSE_gain':checked['gain'],
            'required_gain':checked['gain_min'],'complete_primary':complete,
            'primary_cases':sum(r['observations'] for r in primary),
            'primary_parents':checked['expected_parent_count'],
            'bootstrap_not_a_gate':True,'new_full_wave_states':0,
            'reason':'MATERIAL_GAIN_MET' if status=='PASS' else
                     ('MATERIAL_GAIN_BELOW5_PERCENT' if complete else 'NUMERICAL_PRIMARY_INCOMPLETE'),
            'E1_enabled':status=='PASS',
            'target_access':'only after all32 online prediction artifacts saved',
            'coverage_details':checked}

def run_online(root):
    root=Path(root); book=Book(root,'e0'); cfg=book.config
    if not (root/'results/a32/PREFLIGHT.json').is_file():raise ValueError('PREFLIGHT_REQUIRED')
    online=root/'results/a32/online'; online.mkdir(parents=True,exist_ok=True)
    cases=data.online_cases(cfg); td,meta=data.train_deltas(cfg)
    with book.span('TRAIN_material_basis_freeze',TRAIN_parents=64):
        model=charts.prepare_charts(td)
    freeze={'status':'FROZEN_BEFORE_DEV_TRUTH','config':cfg,'TRAIN':meta,
            'primary':'pca64_calibrated_L_gaussian','hyperparameter_selection':'NONE_ON_DEVELOPMENT',
            'representations':'fixed TRAIN PCA64; first A30 N3-support Sparse32/64'}
    (root/'results/a32/FROZEN_PROTOCOL.json').write_text(json.dumps(freeze,indent=2,ensure_ascii=False)+'\n')
    records=[]; paths=[]
    for i,case in enumerate(cases):
        book.check(); started=time.perf_counter(); bg=case.background
        saved=data.cached_n3(case); noise=np.sqrt(2)/case.sigma
        y=pack_measurement(case.observation-bg['yb'])*noise
        # fcal(c)=F(N3)+L(bg+Tc)-L(N3), with the SAME fixed class.
        offset=pack_measurement(saved.data-bg['yb'])-bg['J']@(np.sqrt(1/576)*material_real(case.n3-bg['chi_b']))
        calibrated_target=y-noise*offset
        arrays={'case_id':np.asarray(case.case_id),'chi_N3':case.n3,'background':bg['chi_b']}
        percase=[]
        records.append({'case_id':case.case_id,'parent_id':case.parent_id,'background_id':case.background_id,
                        'family':case.family,'method':'N3','status':'COMPLETE','success':True,'online_wall_seconds':0.})
        for rep in cfg['representations']:
            with book.span('fixed_material_chart',material_chart_builds=1):
                chart=charts.freeze_chart(model,rep,case.n3,bg['chi_b'],amplitude=1.2)
            T=chart.T; M=chart.material_metric
            target=T.T@((1/576)*material_real(case.n3-bg['chi_b']))
            lower=np.r_[-.5-bg['chi_b'].real,-bg['chi_b'].imag]
            upper=np.r_[4.-bg['chi_b'].real,.5-bg['chi_b'].imag]
            with book.span('box_feasible_N3_chart_projection',coefficient_dimension=len(M)):
                projection=solve_qp(M,target,T,lower,upper,initial=chart.frozen_feasible_initial,config=cfg['solver'],check=book.check)
            c0=_c(projection); projected=chart.decode(c0)
            arrays['chi_'+rep+'_projected_N3']=projected
            arrays['T_'+rep]=T; arrays['M_'+rep]=M; arrays['c0_'+rep]=c0
            pname=rep+'_projected_N3'
            base={'case_id':case.case_id,'parent_id':case.parent_id,'background_id':case.background_id,'family':case.family}
            record=dict(base,method=pname,**{k:clean(projection.get(k)) for k in ['status','success','iterations','kkt_relative','feasibility','objective']},online_wall_seconds=float(projection.get('wall_seconds',0)))
            records.append(record); percase.append(record)
            if not projection['success']: book.failure('PROJECTION_QP',projection.get('status'),case_id=case.case_id,representation=rep)
            with book.span('TRAIN_only_coefficient_priors',coefficient_prior_builds=3):
                priors=build_priors(chart.train_coefficients,T,chart.train_mass,haar_analysis=model.haar_analysis,cell_volume=1/576)
            A=noise*(bg['J']@(np.sqrt(1/576)*T))
            arrays['A_'+rep]=A
            variants=[('L','gaussian',y),('calibrated_L','gaussian',calibrated_target),
                      ('calibrated_L','ridge',calibrated_target),('calibrated_L','multiscale',calibrated_target)]
            for physical,prior_id,target_data in variants:
                method=f'{rep}_{physical}_{prior_id}'; prior=priors[prior_id]
                mean=np.asarray(prior['mean']); P=np.asarray(prior['precision'])
                H=A.T@A+P; h=A.T@target_data+P@mean
                t=time.perf_counter()
                with book.span('continuous_material_QP',material_QP_solves=1):
                    result=solve_qp(H,h,T,lower,upper,initial=c0,config=cfg['solver'],check=book.check)
                elapsed=time.perf_counter()-t; c=_c(result); image=chart.decode(c)
                arrays['chi_'+method]=image; arrays['c_'+method]=c
                if prior_id=='gaussian':
                    arrays['prior_mean_'+rep]=mean; arrays['prior_precision_'+rep]=P
                r=dict(base,method=method,**{k:clean(result.get(k)) for k in ['status','success','iterations','kkt_relative','feasibility','objective']},
                       online_wall_seconds=elapsed,coefficient_dimension=len(c),
                       linear_data_residual=float(la.norm(A@c-target_data)),
                       calibrated_offset_noise_norm=float(la.norm(noise*offset)),
                       prior_metadata=clean(prior.get('metadata',{})))
                records.append(r);percase.append(r)
                if not result['success']:book.failure('ONLINE_QP',result.get('status'),case_id=case.case_id,method=method,kkt=r['kkt_relative'])
        path=online/(case.case_id+'.npz');np.savez_compressed(path,**arrays); paths.append(path.name)
        (online/(case.case_id+'.json')).write_text(json.dumps(clean(percase),indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({'progress':i+1,'total':len(cases),'case_id':case.case_id,
                          'seconds':time.perf_counter()-started,'failed':sum(not bool(x.get('success')) for x in percase)}),flush=True)
    write_csv(online/'ONLINE_METHODS.csv',[{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else v) for k,v in r.items()} for r in records])
    token={'status':'ONLINE_COMPLETE','case_ids':[c.case_id for c in cases],'prediction_files':paths,
           'targets_loaded_before_completion':False,'new_physics_states':0,'methods_frozen':True}
    (online/'ONLINE_COMPLETE.json').write_text(json.dumps(token,indent=2)+'\n')
    return records

def score(root):
    root=Path(root);book=Book(root,'e0-score');cfg=book.config;online=root/'results/a32/online'
    cases=data.online_cases(cfg)
    targets=data.offline_targets(cfg,[x.case_id for x in cases],completion_token_path=online/'ONLINE_COMPLETE.json')
    target_map={str(c):np.asarray(t) for c,t in zip(targets.case_ids,targets.truth)}
    from .frozen_metrics import image_metrics
    from .frozen_representations import digital_topology
    from .statistics import e0_statistics
    out=root/'results/a32/offline';out.mkdir(parents=True,exist_ok=True)
    rows=[];recons={'case_id':np.asarray([x.case_id for x in cases])}
    method_images={}; truth_images=[]; backgrounds=[]
    for case in cases:
        book.check()
        truth=target_map[case.case_id]
        truth=np.asarray(truth,np.complex128);truth_images.append(truth);backgrounds.append(case.background['chi_b'])
        metadata={x['method']:x for x in json.loads((online/(case.case_id+'.json')).read_text())}
        metadata['N3']={'status':'COMPLETE','success':True,'online_wall_seconds':0.}
        with np.load(online/(case.case_id+'.npz'),allow_pickle=False) as z:
            for key in z.files:
                if not key.startswith('chi_'):continue
                method=key[4:];image=z[key].copy();method_images.setdefault(method,[]).append(image)
                met=image_metrics(image,truth,case.background['chi_b'])
                delta=truth-case.background['chi_b'];pd=image-case.background['chi_b'];threshold=.25*np.max(abs(delta))
                topo=digital_topology(abs(pd).reshape(24,24)>=threshold)
                tt=digital_topology(abs(delta).reshape(24,24)>=threshold)
                row={'case_id':case.case_id,'parent_id':case.parent_id,'background_id':case.background_id,'family':case.family,
                     'method':method,**{k:v for k,v in metadata[method].items() if k in ['status','success','online_wall_seconds','kkt_relative','feasibility','iterations']},
                     'fixed_mass_squared_error':float(np.mean(abs(image-truth)**2)),**met,
                     'components':topo.get('components'), 'holes':topo.get('holes'),
                     'component_count_error':abs(topo.get('components',0)-tt.get('components',0)),
                     'hole_count_error':abs(topo.get('holes',0)-tt.get('holes',0))}
                rows.append(row)
            # Independent fixed-chart truth fit, after all ONLINE outputs existed.
            for rep in cfg['representations']:
                T=z['T_'+rep];M=z['M_'+rep];c0=z['c0_'+rep]
                bg=case.background['chi_b']; h=T.T@((1/576)*material_real(truth-bg))
                lower=np.r_[-.5-bg.real,-bg.imag];upper=np.r_[4-bg.real,.5-bg.imag]
                result=solve_qp(M,h,T,lower,upper,initial=c0,config=cfg['solver'],check=book.check)
                image=bg+(T@_c(result))[:576]+1j*(T@_c(result))[576:]
                method=rep+'_fixed_chart_OFFLINE_capacity';method_images.setdefault(method,[]).append(image)
                topo=digital_topology(abs(image-bg).reshape(24,24)>=threshold)
                rows.append({'case_id':case.case_id,'parent_id':case.parent_id,'background_id':case.background_id,'family':case.family,
                             'method':method,'status':result['status'],'success':result['success'],
                             'kkt_relative':result.get('kkt_relative'),'feasibility':result.get('feasibility'),
                             'fixed_mass_squared_error':float(np.mean(abs(image-truth)**2)),
                             **image_metrics(image,truth,bg),'components':topo.get('components'),'holes':topo.get('holes'),
                             'component_count_error':abs(topo.get('components',0)-tt.get('components',0)),
                             'hole_count_error':abs(topo.get('holes',0)-tt.get('holes',0)),
                             'scope':'OFFLINE_TRUTH_FIT_FIXED_ONLINE_SUPPORT'})
    write_csv(out/'E0_CASE_METRICS.csv',rows)
    recons.update(truth=np.asarray(truth_images),background=np.asarray(backgrounds),**{k:np.asarray(v) for k,v in method_images.items()})
    np.savez_compressed(out/'RECONSTRUCTIONS.npz',**recons)
    result=e0_statistics(rows,cfg)
    (out/'E0_SUMMARY.json').write_text(json.dumps(clean(result),indent=2,ensure_ascii=False)+'\n')
    # Exactly the statistic's parent/background completeness rule, never row count alone.
    gate=gate_from_statistics(result,cfg)
    (root/'results/a32/E0_GATE.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps(gate));return gate
