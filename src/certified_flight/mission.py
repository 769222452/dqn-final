"""Mission definition: initial state, path limits, terminal gates and reward.

The configuration file is the single source of these values. Cross-checks make
sure the mission never relaxes the vehicle or environment limits it builds on.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from morphing_waverider.dynamics.atmosphere import Atmosphere
from morphing_waverider.dynamics.point_mass_3dof import PointMass3DOF, VehicleParams

FAILURE_OUTCOMES = ("CONSTRAINT_VIOLATION", "AERO_DOMAIN_EXIT", "NUMERICAL_FAILURE")


class Mission:
    def __init__(self, root: Path, config="configs/mission.yaml"):
        self.root = Path(root).resolve(strict=True)
        self.config_path = self.path(config)
        self.data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self._validate()

    def path(self, rel):
        """Resolve a repository-relative path; refuse paths outside the repository."""
        candidate = (self.root / rel).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"Path escapes the repository: {rel}")
        return candidate

    def _validate(self):
        d = self.data
        if d["schema"] != "mission/1" or d["duration_s"] != 1200:
            raise ValueError("Unexpected mission schema/duration")
        if round(d["duration_s"] / d["decision_dt_s"]) != d["duration_s"] / d["decision_dt_s"]:
            raise ValueError("Nonintegral number of decisions")
        if round(d["decision_dt_s"] / d["plant_dt_s"]) != d["decision_dt_s"] / d["plant_dt_s"]:
            raise ValueError("Nonintegral number of plant steps")
        self.n_decisions = int(d["duration_s"] / d["decision_dt_s"])
        self.n_steps = int(d["duration_s"] / d["plant_dt_s"])
        initial = json.loads(self.path(d["initial_state"]["source"]).read_text(encoding="utf-8"))
        if initial["seed"] != d["initial_state"]["seed"] or \
           initial["x_init_flown"] != d["initial_state"]["x_h_V_gamma_s"] or \
           initial["configured_nominal_x0"] != d["initial_state"]["nominal_unperturbed_x"] or \
           initial["alpha0_rad"] != d["initial_state"]["alpha_rad"]:
            raise ValueError("Mission initial state disagrees with the recorded initial state")
        self.environment = yaml.safe_load(self.path(d["model"]["atmosphere"]).read_text(encoding="utf-8"))
        self.vehicle = yaml.safe_load(
            self.path(d["model"]["vehicle_limits_and_heat_proxy"]).read_text(encoding="utf-8"))
        profile = json.loads(self.path(d["model"]["teacher_dynamic_profile"]).read_text(encoding="utf-8"))
        if profile["mass"]["m_kg"] != d["model"]["mass_kg"] or \
           profile["reference"]["area_m2"] != d["model"]["reference_area_m2"] or \
           self.vehicle["mass"]["m"] != d["model"]["mass_kg"]:
            raise ValueError("Reference area/mass of profile, vehicle and mission disagree")
        lim = self.vehicle["limits"]
        dlim = d["limits"]
        for key, low, high in (("altitude_m", "altitude_min", "altitude_max"),
                               ("velocity_mps", "velocity_min", "velocity_max"),
                               ("gamma_rad", "gamma_min", "gamma_max"),
                               ("alpha_rad", "alpha_min", "alpha_max")):
            if float(dlim[key][1]) != float(lim[high]) or float(dlim[key][0]) < float(lim[low]):
                raise ValueError(f"Mission may not relax the vehicle limit: {key}")
        for key, limit_key in (("dynamic_pressure_pa_max", "q_dyn_max"),
                               ("heat_proxy_wm2_max", "heat_flux_max"),
                               ("load_factor_g_max", "n_load_max"),
                               ("alpha_rate_radps", "alpha_rate_max")):
            if float(dlim[key]) != float(lim[limit_key]):
                raise ValueError(f"Mission must retain the vehicle limit: {key}")
        if dlim["altitude_m"][0] != self.environment["gym_env"]["terminal_altitude"] or \
           d["terminal"]["altitude_m_min"] != dlim["altitude_m"][0] or \
           d["terminal"]["velocity_mps_min"] != dlim["velocity_mps"][0]:
            raise ValueError("Plant and terminal constraints differ")
        if d["terminal"]["required_downrange_m_min"] != self.environment["mission"]["required_downrange_m"]:
            raise ValueError("Mission range gate differs from the environment configuration")
        if d["model"]["support_mach"] != [3.5, 9.25] or d["model"]["support_alpha_deg"] != [-2.0, 12.0]:
            raise ValueError("Aerodynamic table grid is not this support range")
        if not self.environment["dynamics"]["spherical_earth"] or not d["model"]["spherical_earth"]:
            raise ValueError("Spherical-earth dynamics required")

    def model(self, aero):
        """Point-mass 3-DOF model with fixed reference area and the given aerodynamic provider."""
        d = self.data["model"]
        heat = self.vehicle["heat_flux_placeholder"]
        s = d["reference_area_m2"]
        params = VehicleParams(m=d["mass_kg"], S_ref=s, S0=s, k_b=0.0, S_min=s, b_nominal=0.0,
                               span_index=-1, heat_C=float(heat["C"]), heat_R_n=float(heat["R_n"]))
        return PointMass3DOF(Atmosphere(self.environment["atmosphere"]), aero, params, None, True)

    def outcome(self, t, x, reason=None):
        if reason is not None:
            if reason not in FAILURE_OUTCOMES + ("EXTERNAL_INTERRUPT",):
                raise ValueError(reason)
            return reason
        if t < self.data["duration_s"] - 1e-8:
            return "INCOMPLETE"
        if t > self.data["duration_s"] + 1e-8:
            return "NUMERICAL_FAILURE"
        if (x[0] < self.data["terminal"]["altitude_m_min"] or
                x[1] < self.data["terminal"]["velocity_mps_min"] or
                x[3] - self.data["initial_state"]["x_h_V_gamma_s"][3] <
                self.data["terminal"]["required_downrange_m_min"]):
            return "CONSTRAINT_VIOLATION"
        return "SUCCESS"

    def reward(self, ds, outcome="INCOMPLETE"):
        """Downrange progress plus a terminal bonus or penalty."""
        cfg = self.data["reward_interface"]
        progress = cfg["progress_weight"] * ds / cfg["downrange_scale_m"]
        terminal = (cfg["success_bonus"] if outcome == "SUCCESS" else
                    cfg["failure_penalty"] if outcome in FAILURE_OUTCOMES else 0.0)
        terminated = outcome in ("SUCCESS",) + FAILURE_OUTCOMES
        truncated = outcome == "EXTERNAL_INTERRUPT"
        bootstrap = truncated and cfg["bootstrap_on_external_interrupt"]
        return {"reward": progress + terminal, "progress": progress, "terminal": terminal,
                "terminated": terminated, "truncated": truncated, "bootstrap": bootstrap}
