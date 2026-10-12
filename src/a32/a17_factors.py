"""A17 bounded dense algebra adapter. CPU complex128 / real float64.

Input B has shape (P,n,p), mapping ONE shared real material vector to each
illumination's complex current. S can be (m,n) or (P,m,n). Real data order is
per illumination [Re(data); Im(data)], matching private A16 fast_contractions.
No new solver, preconditioner, jitter, full-wave interface or benchmark.
"""
from dataclasses import dataclass
import numpy as np
from scipy import linalg as la
RTOL = 1e-10


def real(X):
    if np.iscomplexobj(X):
        raise ValueError('shared material/data coordinates must be real')
    X = np.asarray(X, dtype=np.float64)
    if not np.all(np.isfinite(X)): raise ValueError('nonfinite real input')
    return X


def orth(X, against=None, rank=None, rtol=RTOL):
    X = np.asarray(X, dtype=np.complex128)
    if X.ndim == 1: X = X[:, None]
    atol = rtol*la.norm(X) if against is not None and against.shape[1] else 0.
    if against is not None and against.shape[1]:
        for _ in range(2): X = X-against@(against.conj().T@X)
    if X.shape[1] == 0: return X.copy()
    u, s, _ = la.svd(X, full_matrices=False)
    k = np.count_nonzero(s > max(atol, rtol*s[0])) if s.size and s[0] else 0
    if rank is not None: k = min(k, rank)
    return np.ascontiguousarray(u[:, :k])


def stability(A, rtol=RTOL):
    s = la.svdvals(A)
    if not s.size: return dict(stable=True, sigma_min=None, sigma_max=None, relative=None, condition=1.)
    ratio = float(s[-1]/s[0]) if s[0] else 0.
    return dict(stable=ratio>rtol, sigma_min=float(s[-1]), sigma_max=float(s[0]),
                relative=ratio, condition=float(s[0]/s[-1]) if s[-1] else float('inf'))


def stable_solve(A, rhs):
    if not stability(A)['stable']: raise ValueError('compressed current near-singular; no jitter')
    return la.solve(A, rhs, assume_a='gen') if A.shape[0] else np.zeros_like(rhs)


def realify(Jc):
    """(P,m,p) complex -> (P*2*m,p) real; shared material p unchanged."""
    Jc = np.asarray(Jc, complex)
    if Jc.ndim == 2: Jc = Jc[None]
    return np.concatenate([Jc.real, Jc.imag], axis=1).reshape(-1, Jc.shape[-1])


def observe(S, currents):
    return np.asarray(S, complex) @ currents


def endpoint(L, S, B, U):
    """Stable Galerkin endpoint; rank-deficient input reduced by orth explicitly."""
    B = np.asarray(B, complex)
    if B.ndim == 2: B = B[None]
    U = orth(U)
    A = U.conj().T@L@U
    info = stability(A)
    if not info['stable']: raise ValueError('endpoint current block unstable')
    coeff = np.stack([stable_solve(A, U.conj().T@b) for b in B])
    K = U@coeff
    Jc = observe(S, K)
    return dict(U=U, actual_rank=U.shape[1], stability=info, K=K, Jc=Jc, J=realify(Jc))


def atom_endpoint(L, S, B, atoms, selected_ids):
    """IDs always index ORIGINAL dictionary, never the orth/SVD basis."""
    ids = tuple(int(i) for i in selected_ids)
    if len(ids)!=len(set(ids)): raise ValueError('duplicate atom identity')
    if any(i<0 or i>=atoms.shape[1] for i in ids): raise ValueError('invalid atom identity')
    out = endpoint(L,S,B,atoms[:,ids])
    out['selected_ids'] = ids
    out['requested_count'] = len(ids)
    return out


@dataclass
class RemovalCore:
    L: np.ndarray
    S: np.ndarray
    B: np.ndarray
    V: np.ndarray
    A: np.ndarray
    old: dict
    stability: dict
    fast_available: bool
    reason: str | None
    K: np.ndarray | None
    J: np.ndarray | None
    factor: tuple | None = None
    removed_cache: dict | None = None


def removal_core(L,S,B,V,A):
    """Prepare ONCE for all incoming banks; singular core keeps direct endpoints."""
    B=np.asarray(B,complex)
    if B.ndim==2: B=B[None]
    V=orth(V); A=orth(A,against=V)
    old=endpoint(L,S,B,np.column_stack([V,A]))
    info=stability(V.conj().T@L@V)
    if not info['stable']:
        return RemovalCore(L,S,B,V,A,old,info,False,'CORE_UNSTABLE_DIRECT_FALLBACK',None,None)
    core=endpoint(L,S,B,V)
    factor=la.lu_factor(V.conj().T@L@V) if V.shape[1] else None
    return RemovalCore(L,S,B,V,A,old,info,True,None,core['K'],core['J'],factor)


def bank_increment(core,Z):
    if not core.fast_available: raise ValueError('fast Schur unavailable')
    Z=orth(Z,against=core.V)
    V,L=core.V,core.L
    Az=V.conj().T@L@Z
    Zbar=Z-V@(la.lu_solve(core.factor,Az) if core.factor is not None else np.zeros_like(Az))
    Gamma=Z.conj().T@L@Zbar
    N=np.stack([Z.conj().T@(b-L@kv) for b,kv in zip(core.B,core.K)])
    delta=np.stack([observe(core.S if np.ndim(core.S)==2 else core.S[i], Zbar@stable_solve(Gamma,n)) for i,n in enumerate(N)])
    return dict(Z=Z,Zbar=Zbar,Gamma=Gamma,N=N,Y=observe(core.S,Zbar),Jc=delta,J=realify(delta))


def swap_from_core(core,C):
    C=orth(C,against=core.V)
    new=endpoint(core.L,core.S,core.B,np.column_stack([core.V,C]))
    direct=new['J']-core.old['J']
    out=dict(old=core.old,new=new,D=direct,fast_available=core.fast_available,
             reason=core.reason,core_stability=core.stability)
    if core.fast_available:
        try:
            if core.removed_cache is None: core.removed_cache=bank_increment(core,core.A)
            a=core.removed_cache; c=bank_increment(core,C)
            out.update(D_schur=c['J']-a['J'],removed=a,added=c)
        except ValueError:
            out.update(fast_available=False,reason='SCHUR_UNSTABLE_DIRECT_FALLBACK')
    return out


def atom_swap(L,S,B,atoms,selected_ids,dropped_ids,added_ids):
    ids=tuple(selected_ids); dropped=tuple(dropped_ids); added=tuple(added_ids)
    if len(set(ids))!=len(ids) or len(set(dropped))!=len(dropped) or len(set(added))!=len(added):
        raise ValueError('duplicate atom IDs')
    if not set(dropped)<=set(ids) or set(added)&set(ids): raise ValueError('invalid move IDs')
    if any(i<0 or i>=atoms.shape[1] for i in ids+dropped+added): raise ValueError('invalid atom identity')
    retained=tuple(i for i in ids if i not in set(dropped))
    core=removal_core(L,S,B,atoms[:,retained],atoms[:,dropped])
    out=swap_from_core(core,atoms[:,added])
    out.update(old_ids=ids,new_ids=retained+added,retained_ids=retained,
               dropped_ids=dropped,added_ids=added)
    if out['old']['actual_rank']!=len(ids) or out['new']['actual_rank']!=len(retained+added):
        out['feasible_fixed_budget']=False; out['rank_reason']='ATOM_DEPENDENCE'
    else: out['feasible_fixed_budget']=len(dropped)==len(added)
    return out


def prior(Lambda,p):
    Lambda=real(Lambda)
    if Lambda.ndim==0: Lambda=np.eye(p)*float(Lambda)
    if Lambda.shape!=(p,p) or not np.allclose(Lambda,Lambda.T,rtol=1e-12,atol=1e-14):
        raise ValueError('Lambda must be symmetric real (p,p)')
    la.cholesky(Lambda,lower=True)
    return Lambda


def gn_step(J,r,Lambda,ell):
    J,r,ell=real(J),real(r),real(ell)
    Lambda=prior(Lambda,J.shape[1]); H=J.T@J+Lambda
    s=la.cho_solve(la.cho_factor(H,lower=True),-(J.T@r+ell))
    return dict(s=s,H=H,normal_residual=la.norm(H@s+J.T@r+ell),Lambda=Lambda)


def exact_step_change(Jo,Jn,r,Lambda,ell):
    o=gn_step(Jo,r,Lambda,ell); n=gn_step(Jn,r,Lambda,ell)
    D=Jn-Jo; x=o['s']; e=r+Jo@x
    hlin=Jo.T@D@x+D.T@e
    h=hlin+D.T@D@x
    d=n['s']-x
    v=-la.solve(o['H'],hlin,assume_a='pos')
    return dict(old=o,new=n,D=D,d=d,v=v,h=h,d_formula=-la.solve(n['H'],h,assume_a='pos'),
                delta_H=Jo.T@D+D.T@Jo+D.T@D)


def paired_support(Jo,D,Ho,*,exact_unconstrained=False):
    """Caller must explicitly attest exact unconstrained common-coordinate endpoints."""
    if not exact_unconstrained: raise ValueError('closure requires exact unconstrained endpoint steps')
    u,s,vt=la.svd(real(D),full_matrices=False)
    q=np.count_nonzero(s>RTOL*s[0]) if s.size and s[0] else 0
    Y=u[:,:q]; Z=s[:q,None]*vt[:q]
    W=np.column_stack([Jo.T@Y,Z.T])
    W=la.solve(Ho,W,assume_a='pos')
    # real orthogonal support uses same scale-aware relative truncation.
    us,ss,_=la.svd(W,full_matrices=False)
    k=np.count_nonzero(ss>RTOL*ss[0]) if ss.size and ss[0] else 0
    return dict(T=us[:,:k],Y=Y,Z=Z,q=int(q),support_rank=int(k))


def objective(J,r,Lambda,ell,s):
    s=real(s); L=prior(Lambda,len(s)); e=real(r)+real(J)@s
    return float(.5*e@e+real(ell)@s+.5*s@L@s)


def gain_directional(uhat,zhat,x,d,Lambda,ell):
    x,d,ell=real(x),real(d),real(ell); L=prior(Lambda,len(x))
    return float(-np.vdot(uhat,zhat).real-(ell+L@x)@d-.5*(np.vdot(zhat,zhat).real+d@L@d))


def four_scores(J,r,Lambda,ell,x,d,v):
    J=real(J); L=prior(Lambda,len(x)); H=J.T@J+L; g=J.T@(r+J@x)+ell+L@x
    return {f'{name}_{kind}':float(-g@w-(.5*w@H@w if kind=='quadratic' else 0.))
            for name,w in [('d',d),('v',v)] for kind in ['linear','quadratic']}


@dataclass(frozen=True)
class OutputErrorBounds:
    eu: float
    ez: float
    provenance: str
    valid_upper_bounds: bool = False
    def __post_init__(self):
        if not self.valid_upper_bounds or not self.provenance.strip():
            raise ValueError('valid output-error upper bounds and provenance required; residual alone insufficient')
        if not np.isfinite(self.eu) or not np.isfinite(self.ez) or min(self.eu,self.ez)<0:
            raise ValueError('finite nonnegative bounds required')


def gain_error_bound(uhat,zhat,bounds):
    if not isinstance(bounds,OutputErrorBounds): raise ValueError('OutputErrorBounds required')
    eu,ez=bounds.eu,bounds.ez
    return float(eu*la.norm(zhat)+(la.norm(uhat)+la.norm(zhat))*ez+eu*ez+.5*ez**2)


def interval(qhat,eps):
    if not np.isfinite(qhat) or not np.isfinite(eps) or eps<0: raise ValueError('invalid interval')
    return dict(qhat=float(qhat),eps=float(eps),lower=float(qhat-eps),upper=float(qhat+eps))


def decide(intervals,*,tau=0.,neighborhood_total,omitted_upper_bound=None,
           gamma=0.,state_costs=None,transition_costs=None,risk_safe=True):
    """All interval rows must be VALID bounds; omitted moves need a VALID common UB."""
    if tau<0 or gamma<0: raise ValueError('nonnegative tau/gamma required')
    if neighborhood_total<len(intervals): raise ValueError('invalid neighborhood coverage')
    state_costs=np.zeros(len(intervals)) if state_costs is None else real(state_costs)
    transition_costs=np.zeros(len(intervals)) if transition_costs is None else real(transition_costs)
    if len(state_costs)!=len(intervals) or len(transition_costs)!=len(intervals): raise ValueError('cost shape mismatch')
    for i,row in enumerate(intervals):
        threshold=gamma*(state_costs[i]+transition_costs[i])+tau
        if row['lower']>threshold and (not risk_safe or row['lower']>=0):
            return dict(status='ACCEPT',index=i,neighborhood_complete=len(intervals)==neighborhood_total)
    complete=len(intervals)==neighborhood_total or omitted_upper_bound is not None
    upper=[row['upper']-gamma*state_costs[i] for i,row in enumerate(intervals)]
    if omitted_upper_bound is not None: upper.append(omitted_upper_bound)
    if complete and max(upper,default=-float('inf'))<=tau:
        return dict(status='CERTIFIED_LOCAL_STOP',index=None,neighborhood_complete=True,
                    certificate_scope='declared finite feasible neighborhood')
    return dict(status='ABSTAIN_UNRESOLVED',index=None,neighborhood_complete=complete)


def current_downdate_inverse(U,L,retained_rank):
    """O1 audit API only, dense reference: no production speed claim."""
    A=U.conj().T@L@U; inv=stable_solve(A,np.eye(A.shape[0]))
    k=retained_rank; E,F,G,H=inv[:k,:k],inv[:k,k:],inv[k:,:k],inv[k:,k:]
    stability_core=stability(A[:k,:k])
    if not stability_core['stable']: raise ValueError('core unavailable')
    return E-F@stable_solve(H,G)


def woodbury_material_inverse(Ho,Jo,D):
    """O4 dense validation only; indefinite middle uses general solve."""
    p=paired_support(Jo,D,Ho,exact_unconstrained=True); Y,Z=p['Y'],p['Z']; q=p['q']
    inv=la.solve(Ho,np.eye(len(Ho)),assume_a='pos')
    if not q: return inv
    W=np.column_stack([Jo.T@Y,Z.T]); A=inv@W
    ki=np.block([[-Y.T@Y,np.eye(q)],[np.eye(q),np.zeros((q,q))]])
    return inv-A@la.solve(ki+W.T@A,A.T,assume_a='gen')
