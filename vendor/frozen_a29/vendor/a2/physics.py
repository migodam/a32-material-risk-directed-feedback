"""Matrix-free 2-D scalar outgoing-Helmholtz VIE for bounded A2 probes.

The convention is ``exp(-i omega t)``, ``g=i H0^(1)/4`` and
``j=chi (e+D j)``.  This module deliberately models a finite square grid;
ordinary Gaussian material terms have nonzero tails on that grid and are not
treated as compactly supported objects.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import math, time
import numpy as np
from scipy.special import hankel1
from scipy.fft import next_fast_len
from scipy.integrate import quad
from scipy.sparse.linalg import LinearOperator, gmres

C0 = 299_792_458.0

def self_cell_green(k: float, h: float) -> complex:
    """Equal-area disk integral of g, including the -1/k**2 endpoint."""
    a = h / math.sqrt(math.pi)
    return .5j * math.pi * a / k * hankel1(1, k*a) - 1.0/(k*k)

def square_self_cell_green(k: float, h: float) -> complex:
    """Adaptive polar integral over the actual square cell, split at corners.

    The radial Hankel integral is analytic; the four angular sectors are
    separately integrated so the square's piecewise radial boundary is not
    replaced by an equal-area disk.
    """
    def radial(R):
        return R*hankel1(1,k*R)/k + 2j/(math.pi*k*k)
    def integrand(theta):
        R=h/(2*max(abs(math.cos(theta)),abs(math.sin(theta))))
        return .25j*radial(R)
    # quadrants remove max() kinks from each adaptive integral.
    vals=[]
    for lo in (-math.pi/4,math.pi/4,3*math.pi/4,5*math.pi/4):
        hi=lo+math.pi/2
        vals.append(quad(lambda t: integrand(t).real,lo,hi,epsabs=2e-12,epsrel=2e-12)[0]
                    +1j*quad(lambda t: integrand(t).imag,lo,hi,epsabs=2e-12,epsrel=2e-12)[0])
    return sum(vals)

@dataclass(frozen=True)
class Geometry:
    n: int = 64
    side: float = .4
    n_tx: int = 8
    n_rx: int = 32
    radius_tx: float = .34
    radius_rx: float = .30
    aperture: str = "full"
    def __post_init__(self):
        if self.n < 8: raise ValueError("n must be >= 8")
        if self.aperture not in {"full", "half"}: raise ValueError("aperture must be full or half")

def grid(g: Geometry):
    h=g.side/g.n; x=-g.side/2+(np.arange(g.n)+.5)*h
    xx,yy=np.meshgrid(x,x,indexing="ij")
    return np.c_[xx.ravel(),yy.ravel()],h

def point_green(dst: np.ndarray, src: np.ndarray, k: float) -> np.ndarray:
    d=np.asarray(dst)[:,None,:]-np.asarray(src)[None,:,:]
    r=np.linalg.norm(d,axis=-1); out=np.zeros(r.shape,complex); m=r>0
    out[m]=.25j*hankel1(0,k*r[m]); return out

class FFTGreen:
    """Toeplitz D with FFT apply; never forms an n^2 by n^2 dense matrix."""
    def __init__(self, geometry: Geometry, k: float, cell_integrated: bool=False, quadrature_order: int=4):
        self.g, self.k=geometry,float(k); self.n=geometry.n; self.npix=self.n*self.n
        self.cell_integrated=bool(cell_integrated); self.quadrature_order=int(quadrature_order)
        if self.quadrature_order < 2: raise ValueError("quadrature_order must be >=2")
        _,self.h=grid(geometry); offs=np.arange(-(self.n-1),self.n)*self.h
        xx,yy=np.meshgrid(offs,offs,indexing="ij"); r=np.hypot(xx,yy)
        ker=np.zeros_like(r,dtype=complex); nz=r>0
        if self.cell_integrated:
            nodes,weights=np.polynomial.legendre.leggauss(self.quadrature_order)
            for a,wa in zip(nodes,weights):
                for b,wb in zip(nodes,weights):
                    rr=np.hypot(xx-a*self.h/2,yy-b*self.h/2); good=rr>0
                    q=np.zeros_like(rr,dtype=complex);q[good]=.25j*hankel1(0,k*rr[good])
                    ker += k*k*(self.h/2)**2*wa*wb*q
            ker[self.n-1,self.n-1]=k*k*square_self_cell_green(k,self.h)
        else:
            ker[nz]=k*k*self.h*self.h*.25j*hankel1(0,k*r[nz])
            ker[self.n-1,self.n-1]=k*k*self_cell_green(k,self.h)
        self.kernel=ker; self.shape=(2*self.n-1,2*self.n-1)
        # Linear convolution needs 3n-2 padding to avoid circular wrap.
        fast=next_fast_len(3*self.n-2); self.fft_shape=(fast,fast)
        self._fk=np.fft.fftn(ker,self.fft_shape)
        # Do not use conj(_fk): that also reverses the uncentred FFT array.
        self._fk_adj=np.fft.fftn(np.conj(ker),self.fft_shape)
    def _apply(self, x: np.ndarray, adjoint=False) -> np.ndarray:
        x=np.asarray(x,complex); was_vector=x.ndim==1
        if was_vector:x=x[:,None]
        if x.shape[0]!=self.npix: raise ValueError("leading dimension must equal n*n")
        y=np.empty_like(x)
        ker_fft=self._fk_adj if adjoint else self._fk
        for j in range(x.shape[1]):
            z=np.fft.ifftn(np.fft.fftn(x[:,j].reshape(self.n,self.n),self.fft_shape)*ker_fft)
            y[:,j]=z[self.n-1:2*self.n-1,self.n-1:2*self.n-1].ravel()
        return y[:,0] if was_vector else y
    def block(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Direct finite Toeplitz lookup for a small dense subblock."""
        rows=np.asarray(rows,int); cols=np.asarray(cols,int)
        ri,rj=np.divmod(rows,self.n); ci,cj=np.divmod(cols,self.n)
        return self.kernel[(ri[:,None]-ci[None,:])+self.n-1,(rj[:,None]-cj[None,:])+self.n-1]
    def matvec(self,x): return self._apply(x)
    def matmat(self,x): return self._apply(x)
    def rmatvec(self,x): return self._apply(x,True)
    def rmatmat(self,x): return self._apply(x,True)
    def aslinearoperator(self):
        return LinearOperator((self.npix,self.npix),matvec=self.matvec,rmatvec=self.rmatvec,
                              matmat=self.matmat,rmatmat=self.rmatmat,dtype=complex)
    def dense(self):
        return self.matmat(np.eye(self.npix,dtype=complex))
    def dense_direct(self):
        """Independent analytic point/self-cell assembly for small certificates."""
        p,_=grid(self.g); d=point_green(p,p,self.k)*self.k**2*self.h**2
        np.fill_diagonal(d,self.k**2*self.kernel[self.n-1,self.n-1]/(self.k**2))
        return d

class BlockJacobi:
    """Ownership-block Jacobi preconditioner for M=I-diag(chi)D.

    Blocks are geometry patches (typically Gaussian component Voronoi cells),
    not a claim that overlapping Gaussian tails are disjoint.
    """
    def __init__(self, D: FFTGreen, chi: np.ndarray, blocks: Iterable[np.ndarray], max_block=256):
        self.indices=[]
        for block in blocks:
            block=np.asarray(block,int)
            self.indices.extend(block[q:q+max_block] for q in range(0,len(block),max_block))
        self.lu=[]
        for ind in self.indices:
            dd=D.block(ind,ind)
            self.lu.append(np.linalg.inv(np.eye(len(ind))-chi[ind,None]*dd))
    def apply(self,r):
        r=np.asarray(r,complex); y=np.zeros_like(r)
        for ind,inv in zip(self.indices,self.lu): y[ind]=inv@r[ind]
        return y
    def operator(self,n): return LinearOperator((n,n),matvec=self.apply,dtype=complex)

class VIE:
    def __init__(self, geometry: Geometry=Geometry(), frequency_hz: float=2.25e9, cell_integrated: bool=False, quadrature_order: int=4):
        self.geometry=geometry; self.points,self.h=grid(geometry); self.k=2*np.pi*frequency_hz/C0
        self.cell_integrated=bool(cell_integrated); self.quadrature_order=quadrature_order
        self.D=FFTGreen(geometry,self.k,cell_integrated,quadrature_order)
        txa=np.linspace(0,2*np.pi,geometry.n_tx,endpoint=False)
        rxa=np.linspace(-np.pi/2,np.pi/2,geometry.n_rx) if geometry.aperture=="half" else np.linspace(0,2*np.pi,geometry.n_rx,endpoint=False)
        self.tx=geometry.radius_tx*np.c_[np.cos(txa),np.sin(txa)]; self.rx=geometry.radius_rx*np.c_[np.cos(rxa),np.sin(rxa)]
        self.E=self._cell_average(self.points,self.tx) if cell_integrated else point_green(self.points,self.tx,self.k)
        self.S=self.k**2*self._cell_integral(self.rx,self.points) if cell_integrated else self.k**2*self.h**2*point_green(self.rx,self.points,self.k)
        self.direct=point_green(self.rx,self.tx,self.k)
    def _cell_integral(self,dst,centres):
        """4x4 Gauss-Legendre source-cell integral of g(dst, source)."""
        nodes,weights=np.polynomial.legendre.leggauss(self.quadrature_order); out=0
        for a,wa in zip(nodes,weights):
            for b,wb in zip(nodes,weights):
                shift=np.array([a,b])*self.h/2
                out=out+wa*wb*(self.h/2)**2*point_green(dst,np.asarray(centres)+shift,self.k)
        return out
    def _cell_average(self,centres,src):
        # Reciprocity makes a target-cell average a source-cell average with
        # arguments swapped/transposed.
        return self._cell_integral(src,centres).T/(self.h*self.h)
    def system(self,chi):
        chi=np.asarray(chi,complex).reshape(-1)
        if len(chi)!=self.D.npix:raise ValueError("chi must have n*n entries")
        return LinearOperator((len(chi),len(chi)),matvec=lambda x:x-chi*self.D.matvec(x),
                              rmatvec=lambda x:x-self.D.rmatvec(np.conj(chi)*x),dtype=complex)
    def solve_linear(self,chi,rhs,rtol=1e-8,maxiter=300,blocks=None):
        """Solve the linear VIE for an already-polarized right-hand side."""
        rhs=np.asarray(rhs,complex); rhs=rhs[:,None] if rhs.ndim==1 else rhs
        A=self.system(chi); pre=BlockJacobi(self.D,np.asarray(chi),blocks).operator(self.D.npix) if blocks is not None else None
        out=np.empty_like(rhs); its=[]
        for q in range(rhs.shape[1]):
            hist=[]; z,info=gmres(A,rhs[:,q],M=pre,rtol=rtol,atol=0,maxiter=maxiter,
                              callback=lambda _:hist.append(1),callback_type="pr_norm")
            if info: raise RuntimeError(f"GMRES did not converge (info={info})")
            out[:,q]=z;its.append(len(hist))
        return out, {"iterations":its,"preconditioner":"block_jacobi" if blocks is not None else "none"}
    def solve(self,chi,rhs=None,rtol=1e-8,maxiter=300,blocks=None):
        """Solve Mj=diag(chi)rhs, for an incident-field right-hand side."""
        rhs=self.E if rhs is None else np.asarray(rhs,complex)
        rhs=rhs[:,None] if rhs.ndim==1 else rhs
        return self.solve_linear(chi,np.asarray(chi)[:,None]*rhs,rtol,maxiter,blocks)
    def forward(self,chi,rtol=1e-8,maxiter=300,blocks=None,return_current=True):
        tic=time.perf_counter(); j,meta=self.solve(chi,rtol=rtol,maxiter=maxiter,blocks=blocks)
        sca=self.S@j; out={"current":j,"scattered":sca,"total":self.direct+sca,"incident":self.direct,
                           "solve":meta,"wall_seconds":time.perf_counter()-tic}
        if not return_current: out.pop("current")
        return out
    def material_tangent(self,chi,dchi,forward=None,rtol=1e-8,blocks=None):
        """Directional derivative: dj=M^-1 diag(dchi)(E+D j)."""
        if forward is None: forward=self.forward(chi,rtol=rtol,blocks=blocks)
        total=self.E+self.D.matmat(forward["current"])
        dj,_=self.solve_linear(chi,np.asarray(dchi)[:,None]*total,rtol=rtol,blocks=blocks)
        return {"current":dj,"scattered":self.S@dj,"total":self.S@dj}
    def dense_reference(self,chi):
        d=self.D.dense_direct(); m=np.eye(self.D.npix)-np.asarray(chi)[:,None]*d
        j=np.linalg.solve(m,np.asarray(chi)[:,None]*self.E)
        return {"current":j,"scattered":self.S@j,"total":self.direct+self.S@j}
