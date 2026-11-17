"""P2 pressure model in numpy: a line-by-line port of datasets/F8D2000/model/p2_fast.m and v7_flow_fast.m.

Strip shock-expansion: every spanwise row of panels is marched from the leading edge; at each panel the local
inclination to the free stream changes by d; d > 0 is an oblique (weak) shock, d < 0 a Prandtl-Meyer expansion,
from the local Mach number of the previous panel. Pressure coefficient Cp = 2 (p/p_inf - 1)/(gamma M_inf^2).
Pressure only; common reference area 6.9319 m^2, length 5 m. A row whose shock detaches (deflection above the
maximum) or that expands to vacuum makes the flight point invalid; so does a streamwise fold-back of the mesh.

The roots (weak shock angle, inverse Prandtl-Meyer) are solved by bisection to machine precision instead of the
safeguarded Newton of the MATLAB file; tests/test_libdesign.py compares with the stored F8D2000 tables.
"""
import numpy as np

G = 1.4
SREF = 6.931920786884801
LREF = 5.
ORIGIN = np.array([2.5, 0., -0.9481139754931033])


def beta_max(M):
    s2 = ((G+1)*M**2/4-1+np.sqrt((G+1)*(1+(G-1)/2*M**2+(G+1)*M**4/16)))/(G*M**2)
    return np.arcsin(np.sqrt(np.clip(s2, 0, 1)))


def theta_of_beta(M, b):
    return np.arctan(2/np.tan(b)*(M**2*np.sin(b)**2-1)/(M**2*(G+np.cos(2*b))+2))


def oblique(M, th):
    """Weak oblique shock: pressure ratio, downstream Mach, ok (attached and supersonic behind)."""
    with np.errstate(all='ignore'):
        lo = np.arcsin(1/M)+1e-13
        hi = beta_max(M)
        maxth = theta_of_beta(M, hi)
        ok = (M > 1) & (th >= 0) & (th <= maxth)
        a, b = lo.copy(), hi.copy()
        for _ in range(22):                      # bracket the weak root
            mid = .5*(a+b)
            f = theta_of_beta(M, mid)-th
            a = np.where(f <= 0, mid, a)
            b = np.where(f > 0, mid, b)
        x = .5*(a+b)
        for _ in range(6):                       # safeguarded Newton (derivative of v7_flow_fast.m)
            A = M**2
            sn, ct = np.sin(x), 1/np.tan(x)
            N = 2*ct*(A*sn**2-1)
            D = A*(G+np.cos(2*x))+2
            f = np.arctan(N/D)-th
            dn = 2*(-(A*sn**2-1)/sn**2+ct*A*np.sin(2*x))
            dd = -2*A*np.sin(2*x)
            df = (dn*D-N*dd)/(D**2+N**2)
            nx = x-f/df
            x = np.where(np.isfinite(nx) & (nx > a) & (nx < b), nx, x)
        beta = np.where(ok, x, np.nan)
        mn1 = M*np.sin(beta)
        mn2 = np.sqrt((1+(G-1)/2*mn1**2)/(G*mn1**2-(G-1)/2))
        pr = 1+2*G/(G+1)*(mn1**2-1)
        M2 = mn2/np.sin(beta-th)
        ok = ok & np.isfinite(M2) & (M2 > 1)
    return pr, M2, ok


def nu(M):
    k = np.sqrt((G+1)/(G-1))
    return k*np.arctan(np.sqrt((M**2-1)/k**2))-np.arctan(np.sqrt(M**2-1))


NU_MAX = np.pi/2*(np.sqrt((G+1)/(G-1))-1)


def nu_inv(v):
    vac = v >= NU_MAX*(1-1e-12)
    capped = v >= nu(1e4)
    a, b = np.zeros_like(v), 4*np.ones_like(v)
    with np.errstate(all='ignore'):
        for _ in range(24):                      # bracket to ~2e-7 in log10(M)
            mid = .5*(a+b)
            f = nu(10**mid)-v
            a = np.where(f <= 0, mid, a)
            b = np.where(f > 0, mid, b)
        q = .5*(a+b)
        for _ in range(6):                       # safeguarded Newton in q = log10(M) to machine precision
            m = 10**q
            f = nu(m)-v
            df = np.log(10)*np.sqrt(m*m-1)/(1+(G-1)/2*m*m)
            nq = q-f/df
            q = np.where(np.isfinite(nq) & (nq > a) & (nq < b), nq, q)
    q = np.where(v <= 0, 0., np.where(capped, 4., q))
    M = 10**q
    M = np.where(vac, np.inf, M)
    M = np.where(v < 0, np.nan, M)
    return M, vac


def expand(M, dth):
    with np.errstate(all='ignore'):
        M2, vac = nu_inv(nu(M)+dth)
        t = (1+(G-1)/2*M**2)/(1+(G-1)/2*M2**2)
        t = np.where(vac, 0., t)
    return t**(G/(G-1)), M2, vac


def skin(native, sgn, Ms, alpha):
    """Forces of one surface. native [nrow, ncol, 3]; Ms, alpha arrays of flight points."""
    P = native*np.array([1., -1., -1.])
    p1, p2, p3, p4 = P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]
    v = sgn*np.cross(p3-p1, p2-p4)
    mag = np.sqrt((v**2).sum(-1))
    if not np.all(mag > 1e-14):
        raise ValueError('degenerate panel')
    normal = v/mag[..., None]
    area = mag/2
    centre = (p1+p2+p3+p4)/4
    nr, nc = area.shape
    nf = len(Ms)
    mid = (native[:-1]+native[1:])/2
    foldback = bool(np.any(np.diff(mid[:, :, 0], axis=1) < -1e-12))
    Ml = np.repeat(Ms[None, :], nr, 0)
    pl = np.ones((nr, nf))
    prev = np.zeros((nr, nf))
    dead = np.zeros((nr, nf), bool)
    vc = np.zeros((nr, nf), bool)
    Fx, Fz, Cm, bad_area, vac_area = (np.zeros(nf) for _ in range(5))
    ca, sa = np.cos(alpha), np.sin(alpha)
    for j in range(nc):
        th = np.arcsin(np.clip(-normal[:, j, 0:1]*ca-normal[:, j, 2:3]*sa, -1, 1))
        d = th-prev
        live = ~dead & ~vc
        c = live & (d > 1e-12)
        e = live & (d < -1e-12)
        kpr = np.ones((nr, nf))
        M2 = Ml.copy()
        if c.any():
            pr, m2, ok = oblique(Ml[c], d[c])
            kpr[c], M2[c] = pr, m2
            dc = dead[c]
            dc[~ok] = True
            dead[c] = dc
        if e.any():
            pr, m2, vac = expand(Ml[e], -d[e])
            kpr[e], M2[e] = pr, m2
            ve = vc[e]
            ve[vac] = True
            vc[e] = ve
        pl = pl*kpr
        Ml = M2
        prev = th
        bad = dead | ~np.isfinite(pl) | ~np.isfinite(Ml)
        Cp = 2*(pl-1)/(1.4*Ms[None, :]**2)
        dFx = -Cp*(area[:, j]*normal[:, j, 0])[:, None]/SREF
        dFz = -Cp*(area[:, j]*normal[:, j, 2])[:, None]/SREF
        Fx = Fx+dFx.sum(0)
        Fz = Fz+dFz.sum(0)
        Cm = Cm+((centre[:, j, 2:3]-ORIGIN[2])*dFx-(centre[:, j, 0:1]-ORIGIN[0])*dFz).sum(0)/LREF
        bad_area = bad_area+(bad*area[:, j:j+1]).sum(0)
        vac_area = vac_area+(vc*area[:, j:j+1]).sum(0)
    valid = ~np.any(dead | vc | ~np.isfinite(pl) | ~np.isfinite(Ml), 0) & (not foldback)
    return dict(Fx=Fx, Fz=Fz, Cm=Cm, valid=valid, bad_area=bad_area, vac_area=vac_area, area=area.sum())


def p2(lower, upper, Ms, alphas):
    """CL, CD_pressure, Cm_pressure, valid for flight points (Ms[k], alphas[k]); lower/upper [50, 50, 3] native."""
    Ms = np.asarray(Ms, float).ravel()
    alphas = np.asarray(alphas, float).ravel()
    u = skin(upper, 1, Ms, alphas)
    l = skin(lower, -1, Ms, alphas)
    Fx = 2*(u['Fx']+l['Fx'])
    Fz = 2*(u['Fz']+l['Fz'])
    Cm = 2*(u['Cm']+l['Cm'])
    valid = u['valid'] & l['valid']
    CL = -Fx*np.sin(alphas)+Fz*np.cos(alphas)
    CD = Fx*np.cos(alphas)+Fz*np.sin(alphas)
    nanif = lambda a: np.where(valid, a, np.nan)
    return dict(CL=nanif(CL), CD=nanif(CD), Cm=nanif(Cm), valid=valid,
                detached=(u['bad_area']+l['bad_area'])/(u['area']+l['area']),
                vacuum=(u['vac_area']+l['vac_area'])/(u['area']+l['area']))


MACH_GRID = np.array([3.5, 4, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25])
ALPHA_GRID_DEG = np.array([-1, 0, 1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 11, 12.])


def grid_table(lower, upper, machs=MACH_GRID, alphas_deg=ALPHA_GRID_DEG):
    """[Mach, alpha] tables on the library grid."""
    MM, AA = np.meshgrid(machs, np.deg2rad(alphas_deg), indexing='ij')
    r = p2(lower, upper, MM.ravel(), AA.ravel())
    shape = MM.shape
    return {k: np.asarray(v).reshape(shape) for k, v in r.items()}
