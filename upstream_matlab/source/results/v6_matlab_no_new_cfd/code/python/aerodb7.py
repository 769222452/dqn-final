"""Queryable aero database for the 1200 s environment, built from MATLAB offdesign_aero_v7 (P2 strip shock-expansion + nominal turbulent friction).
Branches: STRICT (xi in [0,1] along the existing G42->MD9.873 path, 9 nodes) and AREA (lambda 0.90..1.10, xi=(lambda-0.9)/0.2, 9 nodes).
Pressure/friction/base components are kept separate. Out-of-support queries raise DomainExit (never clipped or extrapolated)."""
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import PchipInterpolator
sys.dont_write_bytecode = True
OUT = Path(str((Path(__file__).resolve().parents[4] / 'results/v6_matlab_no_new_cfd')))
class DomainExit(ValueError): pass
def load_branch(branch, prefix="g"):
    import glob
    L = []
    for p in glob.glob(str(OUT/"db"/f"{prefix}_*.mat")):
        m = loadmat(p, squeeze_me=True)
        if str(m["branch"]) == branch: L.append(m)
    L.sort(key=lambda m: float(m["param"]))
    return L
def stack(L, k): return np.stack([np.asarray(m[k], float) for m in L])
class AeroDB7:
    """base: 'none' | 0.5 | 0.3 (p_b/p_inf sensitivity scenarios, NOT validated). friction: True/False. h supplied through .h (set by the environment wrapper)."""
    def __init__(self, branch, base="none", friction=True, d_xi=0.01, d_m=0.05, d_a=0.25, cd_scale=1.0, lam_range=(0.9, 1.1), pressure_grid="refined"):
        L = load_branch(branch); self.branch = branch; self.L = L
        Lp = load_branch(branch, "p") if pressure_grid == "refined" else L; self.Mp, self.Ap = np.asarray(Lp[0]["Mach"], float), np.asarray(Lp[0]["Alpha"], float)
        self.param = np.array([float(m["param"]) for m in L]); self.xi_nodes = (self.param-self.param[0])/(self.param[-1]-self.param[0])
        self.M, self.A, self.H = np.asarray(L[0]["Mach"], float), np.asarray(L[0]["Alpha"], float), np.asarray(L[0]["h_m"], float)
        self.friction = friction; self.base = base; self.cd_scale = cd_scale
        CLp, CDp = stack(Lp, "CLp"), stack(Lp, "CDp")
        CLb, CDb = (np.zeros_like(CLp), np.zeros_like(CDp)) if base == "none" else (stack(Lp, "CLb5" if base == 0.5 else "CLb3"), stack(Lp, "CDb5" if base == 0.5 else "CDb3"))
        self.Cm = stack(Lp, "Cm"); self.valid = np.stack([np.asarray(m["valid"], bool) for m in Lp])
        self.CLp, self.CDp, self.CLb, self.CDb = CLp, CDp, CLb, CDb
        self.CLf, self.CDf = stack(L, "CLf"), stack(L, "CDf")        # [g, M, A, H]
        self.xi_g = np.linspace(0, 1, int(round(1/d_xi))+1); self.m_g = np.arange(self.M[0], self.M[-1]+1e-9, d_m); self.a_g = np.arange(self.A[0], self.A[-1]+1e-9, d_a); self.d = (self.xi_g[1]-self.xi_g[0], d_m, d_a)
        pres = np.stack([self._dense(CLp+CLb, self.Mp, self.Ap), self._dense(CDp+CDb, self.Mp, self.Ap)], -1)
        self.grid = pres
        if friction:   # log-linear in h for the (positive) friction coefficient; dense in (xi,M,alpha) per altitude level
            self.fgrid = np.stack([np.stack([self._dense(self.CLf[..., h], self.M, self.A), self._dense(self.CDf[..., h], self.M, self.A)], -1) for h in range(len(self.H))], 0)
        self.h = None
    def _dense(self, T, M, A):
        t = PchipInterpolator(self.xi_nodes, T, axis=0)(self.xi_g); t = PchipInterpolator(M, t, axis=1)(self.m_g)
        return PchipInterpolator(A, t, axis=2)(self.a_g)
    def _idx(self, v, g, d, name):
        if not (g[0]-1e-9 <= v <= g[-1]+1e-9): raise DomainExit(f"{name}={v:.4f}")
        f = (v-g[0])/d; i = min(int(f), len(g)-2); return i, f-i
    def _tri(self, G, i, fx, j, fm, k, fa):
        c = G[i:i+2, j:j+2, k:k+2]; c = c[0]*(1-fx)+c[1]*fx; c = c[0]*(1-fm)+c[1]*fm; return c[0]*(1-fa)+c[1]*fa
    def components(self, mach, alpha_rad, xi, h=None):
        a = float(alpha_rad)*180.0/np.pi
        i, fx = self._idx(min(max(float(xi), 0.0), 1.0), self.xi_g, self.d[0], "xi"); j, fm = self._idx(float(mach), self.m_g, self.d[1], "Mach"); k, fa = self._idx(a, self.a_g, self.d[2], "alpha_deg")
        p = self._tri(self.grid, i, fx, j, fm, k, fa); out = {"CL_pressure_base": float(p[0]), "CD_pressure_base": float(p[1]), "CL_friction": 0.0, "CD_friction": 0.0}
        if self.friction:
            h = self.h if h is None else h
            if h is None or not (self.H[0]-1e-6 <= h <= self.H[-1]+1e-6): raise DomainExit(f"h={h}")
            hi = min(int(np.searchsorted(self.H, h, side="right")-1), len(self.H)-2); fh = (h-self.H[hi])/(self.H[hi+1]-self.H[hi])
            f0, f1 = (self._tri(self.fgrid[hh], i, fx, j, fm, k, fa) for hh in (hi, hi+1))
            out["CD_friction"] = float(np.exp((1-fh)*np.log(max(f0[1], 1e-12))+fh*np.log(max(f1[1], 1e-12)))); out["CL_friction"] = float((1-fh)*f0[0]+fh*f1[0])
        return out
    def coefficients(self, mach, alpha_rad, eta=None, xi=None):
        xi = float(eta[0]) if xi is None else float(xi); c = self.components(mach, alpha_rad, xi)
        return c["CL_pressure_base"]+c["CL_friction"], (c["CD_pressure_base"]+c["CD_friction"])*self.cd_scale, None
class FixedAero:
    def __init__(self, db, xi): self.db, self.xi = db, float(xi)
    def coefficients(self, mach, alpha_rad, eta=None): return self.db.coefficients(mach, alpha_rad, xi=self.xi)
    @property
    def h(self): return self.db.h
    @h.setter
    def h(self, v): self.db.h = v
