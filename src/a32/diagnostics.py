"""Post-score cache-only failure attribution; never changes the E0 protocol."""
from pathlib import Path
import json
import numpy as np
from scipy import linalg as la
from scipy.optimize import nnls
from .budget import Book
from .data import online_cases,offline_targets,cached_n3
from .preflight import pack_measurement,material_real
from .e0 import write_csv,clean

def run(root):
    root=Path(root); book=Book(root,'diagnostics'); cfg=book.config
    cases=online_cases(cfg); token=root/'results/a32/online/ONLINE_COMPLETE.json'
    targets=offline_targets(cfg,[c.case_id for c in cases],completion_token_path=token)
    truth={str(k):v for k,v in zip(targets.case_ids,targets.truth)}
    validation=Path(cfg['input_root'])/'inputs/a25/validation.npz'
    with np.load(validation,allow_pickle=False) as z:
        if set(z['split'])!={'validation'} or set(map(int,z['parent_id']))!=set(cfg['parent_ids']):
            raise ValueError('EXPOSED_VALIDATION_ONLY_REQUIRED')
        ids=list(map(str,z['case_id'])); clean_data=np.asarray(z['clean']).copy()
        if clean_data.shape!=(128,32,8):raise ValueError('CACHED_CLEAN_LAYOUT_REQUIRED')
    rows=[]; vectors={}
    with np.load(root/'results/a32/offline/RECONSTRUCTIONS.npz',allow_pickle=False) as rec:
        lookup={str(k):i for i,k in enumerate(rec['case_id'])}
        for case in cases:
            book.check(); i=lookup[case.case_id]; bg=case.background; noise=np.sqrt(2)/case.sigma
            true=truth[case.case_id]; yclean=clean_data[ids.index(case.case_id)]
            fanchor=cached_n3(case).data
            Ltrue=bg['yb']+bg['Db']@((true-bg['chi_b'])[:,None]*bg['eb'])
            Lanchor=bg['yb']+bg['Db']@((case.n3-bg['chi_b'])[:,None]*bg['eb'])
            caltrue=fanchor+Ltrue-Lanchor
            with np.load(root/'results/a32/online'/(case.case_id+'.npz'),allow_pickle=False) as z:
                T=z['T_pca64'];M=z['M_pca64'];A=z['A_pca64'];P=z['prior_precision_pca64'];m=z['prior_mean_pca64'];c=z['c_'+cfg['primary_method']]
                x=z['chi_'+cfg['primary_method']]; xproj=z['chi_pca64_projected_N3'];xoracle=rec['pca64_fixed_chart_OFFLINE_capacity'][i]
                co=la.solve(M,T.T@((1/576)*material_real(xoracle-bg['chi_b'])),assume_a='pos')
                target=noise*(pack_measurement(case.observation-fanchor)+pack_measurement(Lanchor-bg['yb']))
                H=A.T@A+P;h=A.T@target+P@m;grad=H@c-h
                lower=np.r_[-.5-bg['chi_b'].real,-bg['chi_b'].imag];upper=np.r_[4-bg['chi_b'].real,.5-bg['chi_b'].imag]
                K=np.vstack((T,-T)); bounds=np.r_[lower,-upper];slack=K@c-bounds
                active=np.flatnonzero(slack<=1e-10)
                mu=nnls(K[active].T,grad,maxiter=200)[0] if len(active) else np.zeros(0)
                normal=-K[active].T@mu if len(active) else np.zeros_like(grad)
                relative_kkt=float(la.norm(grad+normal)/max(la.norm(h),la.norm(H@c),np.finfo(float).tiny))
                C=la.solve(P,np.eye(len(P)),assume_a='pos');covroot=la.cholesky(C,lower=True)
                _,s,vh=la.svd(A@covroot,full_matrices=False);eigen=s*s;weak=eigen<1.
                displacement=T@(c-co);dmass=np.sqrt(1/576)*displacement
                if weak.any():
                    directions=np.sqrt(1/576)*T@covroot@vh[weak].T
                    q,_=la.qr(directions,mode='economic');weak_energy=float(la.norm(q.T@dmass)**2)
                else:weak_energy=0.
                denom=float(dmass@dmass)
                obj=lambda v:float(.5*la.norm(A@v-target)**2+.5*(v-m)@P@(v-m))
                row={'case_id':case.case_id,'parent_id':case.parent_id,'family':case.family,'background_id':case.background_id,
                     'primary_kkt_relative_independent':relative_kkt,'raw_primal_violation':max(0.,float(-slack.min())),
                     'active_normals':int(len(active)),'material_metric_identity_error':float(la.norm(M-1.44*np.eye(len(M)))/la.norm(M)),
                     'background_model_bias_noise_RMS':float(noise*la.norm(pack_measurement(Ltrue-yclean))/np.sqrt(512)),
                     'calibrated_model_bias_noise_RMS':float(noise*la.norm(pack_measurement(caltrue-yclean))/np.sqrt(512)),
                     'N3_full_response_residual_noise_RMS':float(noise*la.norm(pack_measurement(fanchor-case.observation))/np.sqrt(512)),
                     'measurement_noise_RMS':float(noise*la.norm(pack_measurement(case.observation-yclean))/np.sqrt(512)),
                     'prior_whitened_L_eigen_min':float(eigen.min()),'prior_whitened_L_eigen_max':float(eigen.max()),
                     'L_weak_prior_whitened_dimensions':int(weak.sum()),
                     'in_chart_fit_minus_oracle_weak_material_energy_fraction':weak_energy/denom if denom else None,
                     'primary_objective':obj(c),'oracle_material_objective_under_online_model':obj(co),
                     'primary_prior_Mahalanobis_squared':float((c-m)@P@(c-m)),
                     'oracle_prior_Mahalanobis_squared':float((co-m)@P@(co-m)),
                     'pca_projection_MSE':float(np.mean(abs(xproj-true)**2)),
                     'pca_fixed_chart_capacity_MSE':float(np.mean(abs(xoracle-true)**2)),
                     'primary_MSE':float(np.mean(abs(x-true)**2)),
                     'scope':'OFFLINE_CACHE_DIAGNOSIS_NOT_TRUE_POSTERIOR_CERTIFICATE'}
                rows.append(row);vectors[case.case_id+'_eigen']=eigen
                vectors[case.case_id+'_independent_stationarity']=grad+normal
    out=root/'results/a32/offline';write_csv(out/'FAILURE_ATTRIBUTION.csv',rows)
    np.savez_compressed(out/'DIAGNOSTIC_VECTORS.npz',**vectors)
    group={}
    for family in ['complex','shell','asymmetric_piecewise','mixed_scale','smooth']:
        rr=[r for r in rows if (r['family']!='smooth' if family=='complex' else r['family']==family)]
        group[family]={k:float(np.mean([r[k] for r in rr])) for k in
                       ['background_model_bias_noise_RMS','calibrated_model_bias_noise_RMS','measurement_noise_RMS',
                        'L_weak_prior_whitened_dimensions','in_chart_fit_minus_oracle_weak_material_energy_fraction',
                        'primary_prior_Mahalanobis_squared','oracle_prior_Mahalanobis_squared']}
    summary={'status':'COMPLETE_CACHE_ONLY','new_states':0,'new_RHS':0,
             'max_independent_primary_KKT_relative':max(r['primary_kkt_relative_independent'] for r in rows),
             'max_raw_primal_violation':max(r['raw_primal_violation'] for r in rows),'groups':group,
             'scope':'background-L/prior local weak directions; not full-F or calibrated posterior uncertainty',
             'interpretation_limit':'Model bias and TRAIN-prior effects are observed, not uniquely causally separated; no new F at recovered images.'}
    (out/'FAILURE_ATTRIBUTION.json').write_text(json.dumps(clean(summary),indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(clean(summary)));return summary

if __name__=='__main__':run(Path(__file__).resolve().parents[2])
