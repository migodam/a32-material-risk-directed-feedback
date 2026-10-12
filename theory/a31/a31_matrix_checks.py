"""A31 finite-dimensional checks. No mesh Maxwell solver, training, or user data.
Run: OPENBLAS_NUM_THREADS=1 python a31_matrix_checks.py
These are reproducible numerical checks/counterexamples, not recovery evidence.
"""
from __future__ import annotations
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import block_diag
from scipy.optimize import linprog
from scipy.integrate import quad

RNG = np.random.default_rng(311011)
RESULTS: dict[str, dict] = {}

def adj(a): return a.conj().T

def imh(a): return (a-adj(a))/(2j)

def norm(a): return float(np.linalg.norm(a))

def rel(a,b): return norm(a-b)/max(1.,norm(a),norm(b))

def cc(shape): return RNG.normal(size=shape)+1j*RNG.normal(size=shape)

def record(name, passed, **values):
    RESULTS[name] = dict(passed=bool(passed), **values)

def main():
    start=time.perf_counter()
    # 1. Relative feedback and exact finite secant (generic algebra only).
    n=12; s=3
    G=cc((n,n))*.015; inc=cc((n,s))
    a=.3+RNG.random(n)+.15j; h=.08*cc((n,))
    L0=np.eye(n)-a[:,None]*G; L1=np.eye(n)-(a+h)[:,None]*G
    j0=np.linalg.solve(L0,a[:,None]*inc)
    e0=inc+G@j0; j1=np.linalg.solve(L1,(a+h)[:,None]*inc)
    K=np.linalg.solve(L0,h[:,None]*G); rr=np.linalg.solve(L0,h[:,None]*e0)
    v=np.linalg.solve(np.eye(n)-K,rr)
    er=rel(j1-j0,v)
    record('relative_feedback',er<1e-12,relative_error=er)

    # 2. A17 two-sided defect requires generalized-inverse consistency.
    V=np.linalg.qr(cc((n,4)))[0]
    RA=V@np.linalg.solve(adj(V)@L1@V,adj(V))
    RF=np.linalg.inv(L1)
    lhs=RF-RA; rhs=(np.eye(n)-RA@L1)@RF@(np.eye(n)-L1@RA)
    Rbad=.6*RA
    rhsbad=(np.eye(n)-Rbad@L1)@RF@(np.eye(n)-L1@Rbad)
    correction=Rbad-Rbad@L1@Rbad
    er=rel(lhs,rhs); erb=rel(RF-Rbad,rhsbad+correction)
    record('two_sided_defect',max(er,erb)<1e-12,galerkin_error=er,
           arbitrary_inverse_missing_term=norm(RF-Rbad-rhsbad),corrected_error=erb)

    # 3. Signed rank-2r posterior precision update, including nuisance variables.
    m=9; p=7; rank=2
    A=RNG.normal(size=(m,p)); U=RNG.normal(size=(m,rank))*.15
    VV=RNG.normal(size=(p,rank)); A1=A+U@VV.T
    prior=np.diag(1+RNG.random(p)); priorprec=np.linalg.inv(prior)
    H0=priorprec+A.T@A; H1=priorprec+A1.T@A1
    C0=np.linalg.inv(H0); C1=np.linalg.inv(H1)
    Q=np.column_stack((A.T@U,VV))
    Ksmall=np.block([[np.zeros((rank,rank)),np.eye(rank)],
                     [np.eye(rank),U.T@U]])
    pred=C0-C0@Q@np.linalg.solve(np.linalg.inv(Ksmall)+Q.T@C0@Q,Q.T@C0)
    zdim=5
    Heff=H1[:zdim,:zdim]-H1[:zdim,zdim:]@np.linalg.solve(H1[zdim:,zdim:],H1[zdim:,:zdim])
    er=max(rel(pred,C1),rel(np.linalg.inv(Heff),C1[:zdim,:zdim]))
    W=np.diag([1,3,2,1,2,0,0])
    record('signed_covariance_and_nuisance_schur',er<1e-11,relative_error=er,
           signed_risk_change=float(np.trace(W@(C0-C1))),
           precision_difference_eigenvalues=np.linalg.eigvalsh(H1-H0).tolist())

    # 4. Smaller surrogate covariance can mean larger actual material risk.
    jf=.01; ja=.1; ca=1/(1+ja*ja); cf=1/(1+jf*jf)
    ka=ca*ja; kf=cf*jf
    actual=(1-ka*jf)**2+ka*ka
    excess=(ka-kf)**2*(1+jf*jf)
    record('false_confidence_counterexample',ca<cf and actual>cf,
           reported_surrogate_variance=ca,exact_bayes_risk=cf,
           actual_surrogate_risk=actual,excess_identity_error=abs(actual-cf-excess))

    # 5. Passive impedance, physical-form multi-source Gram and Schur balance.
    R=RNG.normal(size=(n,n)); H=(R+R.T)/2
    Dr=RNG.normal(size=(n,5)); Dr=Dr@Dr.T*.05
    Da=np.diag(.2+RNG.random(n)); D=Da+Dr; Z=H-1j*D
    k=5; Azz=Z[:k,:k]; B=Z[:k,k:]; C=Z[k:,:k]; Dz=Z[k:,k:]
    E=np.vstack((np.eye(k),-np.linalg.solve(Dz,C)))
    Zeff=Azz-B@np.linalg.solve(Dz,C)
    er=max(rel(Zeff,adj(E)@Z@E),rel(-imh(Zeff),adj(E)@D@E))
    record('dissipative_schur_congruence',er<1e-12,relative_error=er,
           minimum_reduced_dissipation=float(np.linalg.eigvalsh(-imh(Zeff)).min()))
    Js=np.linalg.solve(Z,inc); gram=(adj(inc)@Js-adj(Js)@inc)/(2j)
    er=rel(gram,adj(Js)@D@Js)
    record('coherent_multisource_power_gram',er<1e-12,relative_error=er)

    # 7. Dual-corrected output certificate without using the inverse in the bound.
    b=cc((n,)); c=cc((n,)); jt=cc((n,))*.1; zt=cc((n,))*.1
    rho=b-Z@jt; tau=c-adj(Z)@zt
    true=np.vdot(c,np.linalg.solve(Z,b)-jt)-np.vdot(zt,rho)
    bound=np.sqrt(np.vdot(rho,np.linalg.solve(Da,rho)).real)*np.sqrt(np.vdot(tau,np.linalg.solve(Da,tau)).real)
    identity=np.vdot(tau,np.linalg.solve(Z,rho))
    er=abs(true-identity)
    record('loss_norm_dual_residual_bound',abs(true)<=bound*(1+1e-12) and er<1e-11,
           absolute_error=float(abs(true)),bound=float(bound),effectivity=float(bound/abs(true)),
           identity_error=float(er))
    Zsharp=-1j*Da; rh=cc((n,)); th=rh.copy()
    val=abs(np.vdot(th,np.linalg.solve(Zsharp,rh)))
    bnd=float(np.vdot(rh,np.linalg.solve(Da,rh)).real)
    record('loss_bound_sharp_case',abs(val-bnd)<1e-10,ratio=float(val/bnd))

    # 9. Oblique projection can make a strictly passive full model active.
    Z2=np.array([[1-1j,10],[10,1-1j]],complex)
    v=np.array([1,0],complex); w=np.array([1,-1j],complex)
    reduced=np.vdot(w,Z2@v)
    record('oblique_passivity_counterexample',np.linalg.eigvalsh(-imh(Z2)).min()>0 and reduced.imag>0,
           full_dissipation_eigenvalues=np.linalg.eigvalsh(-imh(Z2)).tolist(),
           reduced_real=float(reduced.real),reduced_imag=float(reduced.imag),
           reduced_dissipation=float(-reduced.imag),biorthogonality_error=float(abs(np.vdot(w,v)-1)))

    # 10. Spectral radius does not bound a nonnormal resolvent.
    Kn=np.array([[.5,30.],[0.,.5]])
    rn=float(max(abs(np.linalg.eigvals(Kn)))); invnorm=float(np.linalg.norm(np.linalg.inv(np.eye(2)-Kn),2))
    record('nonnormal_resolvent_counterexample',rn<1 and invnorm>100,
           spectral_radius=rn,resolvent_two_norm=invnorm,invalid_radius_only_bound=1/(1-rn))

    # 11. Free complex gain quotient; bipartite rather than interferometric model.
    ff=cc((32,8))+2; gr=np.exp(.1*cc((32,))); hs=np.exp(.1*cc((8,)))
    yy=gr[:,None]*ff*hs[None,:]
    def invariants(x): return x[1:,1:]*x[0,0]/(x[1:,0,None]*x[0,1:][None,:])
    er=rel(invariants(ff),invariants(yy))
    record('bipartite_gain_invariants',er<1e-12,relative_error=er,
           independent_complex_invariants=int((32-1)*(8-1)),complex_orbit_dimension=39)

    # 12. Nuisance integration and signed pair likelihood identity.
    Bn=RNG.normal(size=(9,3)); Gm=np.diag([.1,.2,.3])
    P=np.linalg.inv(np.eye(9)+Bn@Gm@Bn.T)
    a0=RNG.normal(size=9); a1=RNG.normal(size=9); y=RNG.normal(size=9)
    ep0=RNG.normal(size=9)*.1; ep1=RNG.normal(size=9)*.1
    def delta(f,e):
        r=y-f
        return float(r@P@e-.5*e@P@e)
    sec=a1-a0; em=ep1-ep0; ep=(ep1+ep0)/2; rb=y-(a1+a0)/2
    formula=rb@P@em-sec@P@ep-ep@P@em
    er=abs((delta(a1,ep1)-delta(a0,ep0))-formula)
    common=RNG.normal(size=9)
    erc=abs(delta(a1,common)-delta(a0,common)+sec@P@common)
    record('signed_pair_likelihood_defect',max(er,erc)<1e-12,
           identity_error=float(er),shared_offset_identity_error=float(erc),
           shared_offset_changes_log_odds=float(-sec@P@common))

    # 13. Sharp total-variation bound from log-likelihood oscillation.
    osc=1.4; q=np.array([1.,np.exp(-osc/2)]); q/=q.sum()
    delta_vec=np.array([0.,osc]); pp=q*np.exp(delta_vec); pp/=pp.sum()
    tv=.5*float(np.abs(pp-q).sum())
    target=float(np.tanh(osc/4)); muq=float(q[1]); mup=float(pp[1])
    record('sharp_tilt_tv_and_squared_regret',abs(tv-target)<1e-12,
           total_variation=tv,sharp_bound=target,squared_material_regret=(mup-muq)**2)

    # 14. Posterior interval polytope and exact finite-action robust regret LP.
    nn=6; qa=RNG.random(nn); qa/=qa.sum()
    lo=-RNG.random(nn)*1.5; hi=RNG.random(nn)*1.5
    aa=qa*np.exp(lo); bb=qa*np.exp(hi)
    Aub=np.zeros((2*nn,nn+1)); bub=np.zeros(2*nn)
    for i in range(nn):
        Aub[2*i,i]=-1; Aub[2*i,-1]=aa[i]
        Aub[2*i+1,i]=1; Aub[2*i+1,-1]=-bb[i]
    eq=np.r_[np.ones(nn),0.][None,:]
    material=RNG.normal(size=(nn,3))
    loss=((material[:,None,:]-material[None,:,:])**2).sum(axis=2)
    act=int(np.argmin(loss@qa)); robust=0.; solution=None
    for other in range(nn):
        obj=np.r_[-(loss[act]-loss[other]),0.]
        sol=linprog(obj,A_ub=Aub,b_ub=bub,A_eq=eq,b_eq=[1.],bounds=[(0,None)]*(nn+1),method='highs')
        if not sol.success: raise RuntimeError(sol.message)
        value=float(-sol.fun)
        if value>=robust: robust=value; solution=sol.x
    sampled=0.
    for _ in range(3000):
        dd=RNG.uniform(lo,hi); p0=qa*np.exp(dd); p0/=p0.sum()
        sampled=max(sampled,float(loss[act]@p0-min(loss@p0)))
    record('posterior_polytope_material_regret',sampled<=robust+1e-10,
           robust_regret_bound=robust,max_sampled_regret=sampled,
           lp_feasibility_max=float(max(0.,np.max(Aub@solution-bub))))

    # 15. Exact posterior mean transport under exponential tilting.
    dd=RNG.normal(size=nn)*.5; xx=material
    def pt(t):
        u=qa*np.exp(t*dd); return u/u.sum()
    def integrand(t,j):
        u=pt(t); return float(np.sum(u*(xx[:,j]-u@xx[:,j])*dd))
    shift=pt(1)@xx-qa@xx
    transport=np.array([quad(lambda t:integrand(t,j),0,1,epsabs=1e-12)[0] for j in range(3)])
    er=rel(shift,transport)
    record('posterior_mean_covariance_transport',er<1e-10,relative_error=er)

    # 16. Incorrect Dirac reference measure produces fictitious resonance prior.
    magnitudes=np.array([.1,1.])
    unnormalized=1/magnitudes**2; wrong=unnormalized/unnormalized.sum()
    correct=unnormalized*magnitudes**2; correct/=correct.sum()
    record('dirac_jacobian_counterexample',wrong[0]>.99 and rel(correct,[.5,.5])<1e-12,
           wrong_topology_weights=wrong.tolist(),normalized_deterministic_weights=correct.tolist())

    # 17. Exact analytic one-channel Maxwell slab nonidentifiability.
    def slab_transfer(nref):
        phase=2*np.pi*nref
        return np.array([[np.cos(phase),1j*np.sin(phase)/nref],
                         [1j*nref*np.sin(phase),np.cos(phase)]])
    er=rel(slab_transfer(1.),slab_transfer(2.))
    record('analytic_lossless_slab_nonidentifiability',er<1e-12,
           relative_transfer_difference=er,relative_permittivities=[1.,4.],
           thickness_in_vacuum_wavelengths=1.,scope='normal-incidence or single TEM channel; not ring-array imaging')

    # 18. A goal-only compensated separation lower bound can avoid huge norm defects.
    # U=[-1,1], cheap secant=(1+u,2); true response adds (100,0).
    qgoal=np.array([0.,1.]); delta_c=np.array([1.,2.]); Bc=np.array([[1.],[0.]])
    lower=float(qgoal@delta_c-norm(Bc.T@qgoal))
    shared_error=np.array([100.,0.])
    goal_error=float(abs(qgoal@shared_error)); vector_error=norm(shared_error)
    record('goal_only_compensation_certificate',lower==2 and goal_error==0,
           certified_separation_lower=lower,goal_error=goal_error,euclidean_model_error=vector_error,
           scope='linear algebra, not evidence for a physical material family')

    # 19. Mean may violate topology without violating squared-loss optimality.
    images=np.array([[1.,0.,1.],[1.,1.,1.]])
    probs=np.array([.5,.5]); mean=probs@images
    Rmean=float(np.sum(probs*np.sum((images-mean)**2,axis=1)))
    Rsolid=float(np.sum(probs*np.sum((images-images[1])**2,axis=1)))
    record('topology_mixture_mean',Rmean<Rsolid,posterior_mean=mean.tolist(),
           mean_squared_risk=Rmean,solid_action_risk=Rsolid)

    # 20. Reach-controlled representation error is sharper than a triangle bound.
    ep=.2; theta=.8; pp0=np.array([1.,0.]); truth=(1+ep)*pp0
    action=np.array([np.cos(theta),np.sin(theta)])
    chord=norm(action-pp0)**2; loss0=norm(action-truth)**2
    low=ep**2+(1-ep)*chord; high=ep**2+(1+ep)*chord
    record('positive_reach_representation_bound',low-1e-12<=loss0<=high+1e-12,
           material_squared_error=loss0,lower=low,upper=high)

    # 21. Nonlinear ROM derivative has a source-field defect beyond frozen A17.
    mat=a+h; jv=RA@(mat[:,None]*inc); ev=inc+G@jv
    ef=inc+G@j1; direction=cc((n,))*.1
    BF=direction[:,None]*ef; BV=direction[:,None]*ev
    full_der=np.linalg.solve(L1,BF); rom_der=RA@BV
    Ires=(np.eye(n)-L1@RA)@BF
    frozen=(np.eye(n)-RA@L1)@np.linalg.solve(L1,Ires)
    field_term=RA@(direction[:,None]*(G@(j1-jv)))
    er=rel(full_der-rom_der,frozen+field_term)
    record('nonlinear_rom_source_field_defect',er<1e-11,
           relative_identity_error=er,omitted_source_field_term_norm=norm(field_term),
           frozen_only_relative_error=rel(full_der-rom_der,frozen))

    # 22. A genuine tiny radiating-vector-dipole power check, not arbitrary G.
    # Units: epsilon0=1, k=1. Each physical dipole obeys p=alpha_QS E_local,
    # with the universal radiative self term i*k^3/(6*pi) in G_complete.
    positions=np.array([[0.,0.,0.],[0.,0.,.8],[.4,0.,1.6]])
    nd=9; kk=1.; Gphys=np.zeros((nd,nd),complex)
    for ii in range(3):
        Gphys[3*ii:3*ii+3,3*ii:3*ii+3]=1j*kk**3/(6*np.pi)*np.eye(3)
        for jj in range(3):
            if ii==jj: continue
            vec=positions[ii]-positions[jj]; rr0=norm(vec); unit=vec/rr0
            uu=np.outer(unit,unit)
            green=np.exp(1j*kk*rr0)/(4*np.pi*rr0)*(kk**2*(np.eye(3)-uu)+(1/rr0**2-1j*kk/rr0)*(3*uu-np.eye(3)))
            Gphys[3*ii:3*ii+3,3*jj:3*jj+3]=green
    alpha=np.repeat(np.array([2+.2j,3+.3j,1+.5j]),3)
    Zphys=np.diag(1/alpha)-Gphys
    dabs=np.diag(alpha.imag/np.abs(alpha)**2); drad=imh(Gphys)
    ein=np.zeros((nd,2),complex)
    for ii,pos in enumerate(positions):
        ein[3*ii:3*ii+3,0]=np.array([1.,0.,0.])*np.exp(1j*kk*pos[2])
        ein[3*ii:3*ii+3,1]=np.array([0.,1.,0.])*np.exp(1j*kk*pos[0])
    dip=np.linalg.solve(Zphys,ein)
    ext=(adj(ein)@dip-adj(dip)@ein)/(2j)
    absorb=adj(dip)@dabs@dip; radiate=adj(dip)@drad@dip
    er=rel(ext,absorb+radiate)
    minimum=float(np.linalg.eigvalsh(drad).min())
    record('physical_vector_dipole_power_balance',er<1e-11 and minimum>=-1e-12,
           relative_power_identity_error=er,minimum_radiation_gram_eigenvalue=minimum,
           absorption_per_source=np.diag(absorb).real.tolist(),
           radiation_per_source=np.diag(radiate).real.tolist(),
           physical_material_states=1,complex_current_dimension=9,source_rhs=2,
           scope='Radiation-corrected point-dipole Maxwell model, not continuum material imaging or the A17/A24 backend')

    out=dict(schema='A31-finite-checks-v1',seed=311011,
        status='PASS' if all(v['passed'] for v in RESULTS.values()) else 'FAIL',
        checks=len(RESULTS),passed=sum(v['passed'] for v in RESULTS.values()),
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        scope=dict(mesh_maxwell_solves=0,analytic_slab_material_cases=2,tiny_vector_dipole_material_states=1,tiny_vector_dipole_rhs=2,training_updates=0,
                   gpu_calls=0,user_truth_arrays_read=0,
                   note='Floating-point verification and explicit counterexamples; not interval-arithmetic proof or imaging evidence.'),
        elapsed_seconds=time.perf_counter()-start,results=RESULTS)
    path=Path(__file__).with_name('A31_MATRIX_CHECKS.json')
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))
    if out['status']!='PASS': raise SystemExit(1)

if __name__=='__main__': main()
