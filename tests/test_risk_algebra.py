import unittest
import numpy as np
from scipy import linalg as la
from src.a32.risk import paired_feedback,risk_damping,exact_risk_audit,galerkin_source_field_defect,nonlinear_source_defect

class Algebra(unittest.TestCase):
    def test_source_complete_schur_and_signed_pair(self):
        g=np.random.default_rng(321012); n,d,P,m=14,7,4,5
        cc=lambda shape:g.normal(size=shape)+1j*g.normal(size=shape)
        L=np.eye(n)+.015*cc((n,n)); S=cc((m,n)); B=cc((P,n,d))
        Q=la.qr(cc((n,9)),mode='economic')[0]; prior=np.diag(np.linspace(.3,2.,d))
        r=paired_feedback(L,S,B,Q[:,:3],Q[:,3:],prior,h=g.normal(size=d))
        self.assertLess(r['factorization_relative_error'],1e-10)
        self.assertLess(r['signed_H_identity_relative_error'],1e-10)
        self.assertLess(r['paired_mean_range_error'],1e-10)
        self.assertEqual(r['J0'].shape,(P*2*m,d))

    def test_one_hvp_risk_and_bound(self):
        g=np.random.default_rng(321013); p=9
        J=g.normal(size=(16,p)); Jh=J+.12*g.normal(size=J.shape)
        prior=np.diag(np.linspace(.4,1.,p)); H=J.T@J+prior; Hh=Jh.T@Jh+prior
        M=np.diag(np.linspace(.7,2.,p)); h=g.normal(size=p); b=la.solve(Hh,h)
        pair=b+.01*g.normal(size=p)
        called=[]
        result=risk_damping(b,pair,h,M,prior,Hh,lambda q:(called.append(1) or H@q))
        audit=exact_risk_audit(result,b,h,M,H)
        self.assertEqual(len(called),1)
        self.assertLess(audit['risk_identity_absolute_error'],1e-12)
        self.assertLess(audit['bound_violation'],1e-12)

    def test_alpha_zero_retains_baseline_not_initial_state(self):
        b=np.array([.3,-.4]); p=b.copy(); out=risk_damping(b,p,np.ones(2),np.eye(2),np.eye(2),2*np.eye(2),lambda q:None)
        np.testing.assert_array_equal(out['proposal'],b)
        self.assertEqual(out['hvp_calls'],0)

    def test_GN_gain_does_not_imply_material_gain(self):
        H=np.diag([100.,1.]); b=np.array([.1,0.]); c=np.array([0.,.5])
        self.assertLess(c@H@c,b@H@b)
        self.assertGreater(c@c,b@b)
        self.assertAlmostEqual(b@b-c@c,-.24)

    def test_source_field_missing_even_if_observed_state_exact(self):
        G=np.array([[.2j,2],[2,.2j]]); x=np.array([0.,1+.1j]); inc=np.ones((2,1),complex)
        L=np.eye(2)-x[:,None]*G; S=np.array([[1.,0.]],complex); V=np.array([[1.],[0.]],complex)
        R=lambda q:V@la.solve(V.conj().T@L@V,V.conj().T@q)
        j=la.solve(L,x[:,None]*inc); ja=R(x[:,None]*inc)
        ef=inc+G@j; ea=inc+G@ja
        Bf=np.diag(ef[:,0]); Ba=np.diag(ea[:,0])
        out=galerkin_source_field_defect(L,S,Bf,Ba,V,lambda q:la.solve(L,q))
        self.assertLess(la.norm(S@(j-ja)),1e-12)
        self.assertLess(la.norm(out['frozen_channel']),1e-12)
        self.assertGreater(out['omitted_source_field_norm'],1.)
        self.assertLess(out['total_relative_error'],1e-12)

    def test_full_nonlinear_residual_two_channels(self):
        g=np.random.default_rng(321014); n=8; cc=lambda s:g.normal(size=s)+1j*g.normal(size=s)
        G=.03*cc((n,n)); x=.2+.1j+.01*cc((n,)); inc=cc((n,2)); dx=.1*cc((n,))
        L=np.eye(n)-x[:,None]*G; b=x[:,None]*inc; db=dx[:,None]*inc
        V=la.qr(cc((n,3)),mode='economic')[0]; R=lambda q:V@la.solve(V.conj().T@L@V,V.conj().T@q)
        jt=R(b); djt=R(dx[:,None]*(inc+G@jt)); jf=la.solve(L,b)
        true=cc((4,n))
        S=true; exact=S@la.solve(L,dx[:,None]*(inc+G@jf))-S@djt
        out=nonlinear_source_defect(L,S,b,db,dx[:,None]*G,jt,djt,lambda q:la.solve(L,q))
        np.testing.assert_allclose(exact,out['sum'],rtol=1e-11,atol=1e-12)

    def test_damping_is_not_covariance_prior(self):
        J=np.array([[2.,1.],[.1,.2]]); prior=np.diag([.1,.3]); H=J.T@J+prior
        self.assertGreater(la.norm(la.inv(H)-la.inv(H+np.eye(2))),.1)

    def test_same_absolute_prior_after_reanchor(self):
        P=np.diag([2.,3.]); m=np.array([.1,.4]); a=np.array([.2,-.1]); z=np.array([.3,.05])
        abs_loss=(a+z-m)@P@(a+z-m)
        local=(z-(m-a))@P@(z-(m-a))
        self.assertAlmostEqual(abs_loss,local)

if __name__=='__main__':unittest.main()
