"""3-DOF point-mass longitudinal dynamics (vertical plane).

State   x = [h, V, gamma, s]     altitude [m], speed [m/s], flight-path angle [rad], downrange [m]
Control u = [alpha]              angle of attack [rad]  (bank angle sigma reserved for V1+)
Param   eta                      prescribed geometry/blend input to the aerodynamic provider

Equations (non-rotating spherical Earth, zero bank, zero thrust):

    hdot     = V sin(gamma)
    Vdot     = -D/m - g sin(gamma)
    gammadot = L/(m V) - (g/V - V/r) cos(gamma)
    sdot     = V cos(gamma) * R_e / r,          r = R_e + h

Flat-Earth option (spherical_earth: false): g = g0, drop V/r term, sdot = V cos(gamma).

    L = q S_ref CL,   D = q S_ref CD,   q = 0.5 rho V^2

Reference-area convention: L = q S_ref CL and D = q S_ref CD with constant S_ref.
certified_flight.mission constructs VehicleParams directly from mission.yaml.
The geometry_area diagnostic is retained; it is not used to reconstruct forces.

The module depends on an AeroModel only through `coefficients(mach, alpha, eta)`; it
never imports the generator, validator, or RL packages. Every expression is written with
dynamics.backend so it is valid both for numpy floats and CasADi symbols.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..utils.units import G0
from . import backend as B
from .atmosphere import Atmosphere

STATE_NAMES = ("h", "V", "gamma", "s")
CONTROL_NAMES = ("alpha",)
NX = len(STATE_NAMES)
NU = len(CONTROL_NAMES)


class DynamicsError(RuntimeError):
    """Raised on non-finite state / invalid input."""


class CoefficientProvider(Protocol):
    """Minimal aero protocol needed by the dynamics (structural typing)."""

    def coefficients(self, mach, alpha, eta):  # -> (CL, CD, Cm)
        ...


@dataclass(frozen=True)
class VehicleParams:
    m: float
    S_ref: float           # constant aerodynamic reference area [m^2] (SIMULATED, NOT REAL VEHICLE DATA)
    S0: float
    k_b: float
    S_min: float
    b_nominal: float
    span_index: int
    heat_C: float
    heat_R_n: float



class PointMass3DOF:
    def __init__(self, atmosphere: Atmosphere, aero: CoefficientProvider, params: VehicleParams,
                 spec: object | None, spherical_earth: bool = True):
        self.atm = atmosphere
        self.aero = aero
        self.p = params
        self.spec = spec
        self.spherical = bool(spherical_earth)

    # ------------------------------------------------------------- areas
    def reference_area(self, eta=None) -> float:
        """Constant aerodynamic reference area S_ref used in L = q S_ref CL, D = q S_ref CD.
        `eta` is accepted for call-site compatibility and deliberately ignored."""
        return self.p.S_ref

    def geometry_area(self, eta):
        """S_geom(eta): geometry-dependent planform area (diagnostic only, not a force reference)."""
        if self.p.span_index < 0:
            return self.p.S0
        b = eta[self.p.span_index]
        S = self.p.S0 + self.p.k_b * (b - self.p.b_nominal)
        return B.fmax(S, self.p.S_min)

    S_geom = geometry_area

    # ---------------------------------------------------------- quantities
    def path_quantities(self, x, u, eta) -> dict[str, Any]:
        """All auxiliary quantities used by dynamics, constraints and logging."""
        h, V, gamma = x[0], x[1], x[2]
        alpha = u[0]
        rho = self.atm.density(h)
        a = self.atm.speed_of_sound(h)
        mach = V / a
        g = self.atm.gravity(h, spherical=self.spherical)
        q = 0.5 * rho * V**2
        S_ref = self.reference_area()
        S_geom = self.geometry_area(eta)
        CL, CD, Cm = self.aero.coefficients(mach, alpha, eta)
        L = q * S_ref * CL
        D = q * S_ref * CD
        heat_flux = self.p.heat_C * B.sqrt(rho / self.p.heat_R_n) * V**3
        n_load = B.sqrt(L**2 + D**2 + 1e-6) / (self.p.m * G0)
        quantities = {
            "rho": rho, "a": a, "mach": mach, "g": g, "q_dyn": q,
            "S_ref": S_ref, "S_geom": S_geom, "S": S_ref,       # "S" = legacy alias of S_ref
            "CL": CL, "CD": CD, "L": L, "D": D,
            "heat_flux": heat_flux, "n_load": n_load,
        }
        # Pressure-only V5 has no validated pitching-moment model. Absence is
        # preserved; point-mass translation does not require a fictitious Cm=0.
        if Cm is not None:
            quantities["Cm"] = Cm
        return quantities

    # ----------------------------------------------------------------- rhs
    def rhs(self, x, u, eta):
        return self.rhs_with_quantities(x, u, eta)[0]

    def rhs_with_quantities(self, x, u, eta):
        """(x_dot, path_quantities) from one evaluation -- lets stored-state integrators log the
        aerodynamic quantities at a sample without recomputing them."""
        h, V, gamma = x[0], x[1], x[2]
        qty = self.path_quantities(x, u, eta)
        L, D, g = qty["L"], qty["D"], qty["g"]
        m = self.p.m
        sg, cg = B.sin(gamma), B.cos(gamma)
        V_safe = B.fmax(V, 1.0)
        if self.spherical:
            r = self.atm.cfg.R_earth + h
            hdot = V * sg
            Vdot = -D / m - g * sg
            gdot = L / (m * V_safe) - (g / V_safe - V_safe / r) * cg
            sdot = V * cg * self.atm.cfg.R_earth / r
        else:
            hdot = V * sg
            Vdot = -D / m - g * sg
            gdot = L / (m * V_safe) - (g / V_safe) * cg
            sdot = V * cg
        return B.stack([hdot, Vdot, gdot, sdot]), qty

    # ---------------------------------------------------------- validation
    @staticmethod
    def validate_state(x) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.shape[0] != NX:
            raise DynamicsError(f"state must have {NX} entries")
        if not np.all(np.isfinite(arr)):
            raise DynamicsError(f"non-finite state: {arr}")
        if arr[1] <= 0.0:
            raise DynamicsError(f"velocity must be positive: {arr[1]}")
        return arr

    @staticmethod
    def validate_control(u) -> np.ndarray:
        arr = np.asarray(u, dtype=float).reshape(-1)
        if arr.shape[0] != NU:
            raise DynamicsError(f"control must have {NU} entries")
        if not np.all(np.isfinite(arr)):
            raise DynamicsError(f"non-finite control: {arr}")
        return arr

    def numeric_outputs(self, x, u, eta) -> dict[str, float]:
        q = self.path_quantities(np.asarray(x, float), np.asarray(u, float), np.asarray(eta, float))
        return {k: float(v) for k, v in q.items()}
