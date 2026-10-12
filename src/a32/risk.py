"""Source-complete paired material algebra and one-HVP local risk checking.

All posterior statements here are LOCAL_GAUSSIAN_ONLY.  Trust damping and
hard-box normals are never silently put into a posterior covariance.
"""
from __future__ import annotations
import numpy as np
from scipy import linalg as la
from .a17_factors import bank_increment, removal_core, endpoint, realify

def relative(a,b):
    a,b=np.asarray(a),np.asarray(b)
    scale=float(la.norm(b))
    return float(la.norm(a-b)/scale) if scale else (0. if not la.norm(a-b) else None)

def paired_feedback(L,S,B,V,C,precision, *, h=None):
    """Exact retained-to-enlarged Galerkin difference with COMMON injection."""
    core=removal_core(L,S,B,V,np.zeros((len(L),0),complex))
    inc=bank_increment(core,C)
    bigger=endpoint(L,S,B,np.column_stack((core.V,inc['Z'])))
    J0=core.J; J1=J0+inc['J']; D=inc['J']
    H0=J0.T@J0+precision; H1=J1.T@J1+precision
    # Real low-rank D=U Z^T.  This is AFTER all source blocks are lifted.
    u,s,vh=la.svd(D,full_matrices=False)
    rank=int(np.count_nonzero(s>1e-10*s[0])) if len(s) and s[0] else 0
    U=u[:,:rank]*s[:rank]; Z=vh[:rank].T
    Q=np.column_stack((J0.T@U,Z))
    K=np.block([[np.zeros((rank,rank)),np.eye(rank)],[np.eye(rank),U.T@U]])
    closure=la.solve(H0,Q,assume_a='pos')
    C0=la.solve(H0,np.eye(len(H0)),assume_a='pos')
    C1=la.solve(H1,np.eye(len(H1)),assume_a='pos')
    out={'J0':J0,'J1':J1,'D':D,'H0':H0,'H1':H1,'C0':C0,'C1':C1,
         'increment':inc,'retained_rank':core.V.shape[1],'enlarged_rank':bigger['actual_rank'],
         'real_material_increment_rank':rank,'paired_support':closure,
         'factorization_relative_error':relative(J1,bigger['J']),
         'signed_H_identity_relative_error':relative(H1-H0,Q@K@Q.T),
         'signed_H_eigenvalues':la.eigvalsh(H1-H0),
         'posterior_status':'LOCAL_GAUSSIAN_ONLY_UNTRUNCATED',
         'paired_scope':'EXACT_UNCONSTRAINED_COMMON_B_ONLY'}
    if h is not None:
        mu0=C0@h; mu1=C1@h; change=mu1-mu0
        projection=closure@la.lstsq(closure,change)[0] if closure.shape[1] else np.zeros_like(change)
        # Least squares diagnoses range only; no inverse/physics pseudoinverse.
        out.update(mean0=mu0,mean1=mu1,paired_mean_range_error=relative(projection,change))
    return out

def risk_damping(b,paired,h,M,precision,Hhat,hvp):
    b=np.asarray(b,float); d=np.asarray(paired,float)-b
    Md=M@d; quadratic=float(d@Md)
    if quadratic<0: raise ValueError('MATERIAL_METRIC_NOT_SPD')
    if quadratic==0:
        return {'alpha':0.,'proposal':b.copy(),'d':d,'q':np.zeros_like(d),'dual_residual':np.zeros_like(d),
                'Ahat':0.,'Bdual':0.,'gain_hat_full':0.,'gain_lower':0.,'d_M_d':0.,
                'sign_resolved':False,'hvp_calls':0,'status':'ZERO_PAIRED_CHANGE'}
    q=la.solve(Hhat,Md,assume_a='pos')
    Hq=np.asarray(hvp(q),float); rho=Md-Hq
    rhon=float(np.sqrt(max(0.,rho@la.solve(precision,rho,assume_a='pos'))))
    hn=float(np.sqrt(max(0.,h@la.solve(precision,h,assume_a='pos'))))
    Ahat=float(2*q@h-2*d@(M@b)); bound=2*rhon*hn
    alpha=float(np.clip((Ahat-bound)/(2*quadratic),0.,1.))
    return {'alpha':alpha,'proposal':b+alpha*d,'d':d,'q':q,'dual_residual':rho,
            'Ahat':Ahat,'Bdual':bound,'gain_hat_full':Ahat-quadratic,
            'gain_lower':alpha*(Ahat-bound)-alpha*alpha*quadratic,'d_M_d':quadratic,
            'sign_resolved':bool(abs(Ahat-quadratic)>bound),'hvp_calls':1,
            'status':'LOCAL_GAUSSIAN_ONLY','nonlinear_boundary_remainder':'NOT_BOUNDED'}

def exact_risk_audit(result,b,h,M,Hfull):
    mu=la.solve(Hfull,h,assume_a='pos'); d=result['d']; a=result['alpha']
    gain=lambda t:float(2*t*d@(M@(mu-b))-t*t*d@(M@d))
    correction=float(2*result['dual_residual']@mu)
    identity=result['Ahat']-result['d_M_d']+correction
    return {'local_gain_full':gain(1.),'local_gain_damped':gain(a),
            'risk_identity_absolute_error':abs(gain(1.)-identity),
            'bound_violation':max(0.,result['gain_lower']-gain(a)),
            'untruncated_posterior_mean':mu,'scope':'EXACT_LOCAL_GAUSSIAN_NOT_TRUE_MATERIAL_RISK'}

def nonlinear_source_defect(L,S,b,db,dXG,jtilde,djtilde,solve):
    """D(F-Ftilde)=SR D(rho)+SR DX G R rho, tied to actual surrogate."""
    rho=b-L@jtilde
    drho=db+dXG@jtilde-L@djtilde
    state_error=solve(rho)
    derivative_term=S@solve(drho)
    state_feedback_term=S@solve(dXG@state_error)
    return {'rho':rho,'Drho':drho,'derivative_channel':derivative_term,
            'state_feedback_channel':state_feedback_term,'sum':derivative_term+state_feedback_term}

def galerkin_source_field_defect(L,S,Bfull,Brom,V,solve):
    """True fixed-basis nonlinear ROM derivative, not frozen-B derivative."""
    R=lambda x:V@la.solve(V.conj().T@L@V,V.conj().T@x)
    full=S@solve(Bfull); reduced=S@R(Brom)
    injection=Bfull-L@R(Bfull)
    Rinj=solve(injection)
    frozen=S@(Rinj-R(L@Rinj))
    field=S@R(Bfull-Brom)
    return {'full':full,'rom_true_derivative':reduced,'frozen_channel':frozen,
            'source_field_channel':field,'total_relative_error':relative(frozen+field,full-reduced),
            'omitted_source_field_norm':float(la.norm(field))}
