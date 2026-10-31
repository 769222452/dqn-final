"""Legacy conservative CL screen with continuous, audited extreme witnesses.

The outer envelope uses the rate-reachable endpoint bounds of each half-second
slot. Those bounds contain every continuous-alpha trajectory during that slot,
so the original necessary envelope remains conservative. Only its constructive
witnesses change: their first node is the actual current alpha and their future
nodes are joined linearly. A witness is never a complete reachable-set claim.
"""
import numpy as np

from certified_flight.reachability import CLReachability


class _WitnessPlant:
    """Per-call adapter; it stores no state on the shared continuous plant."""

    def __init__(self, plant, alpha):
        self.plant, self.alpha = plant, float(alpha)
        self.db, self.mission = plant.db, plant.mission
        self.atmosphere = plant.atmosphere
        self.normalized_margins = plant.normalized_margins

    def quantities(self, *args):
        base = self.plant if hasattr(self.plant, 'quantities') else self.plant.base
        return base.quantities(*args)

    def integrate(self, x, endpoints, times, source, target, fraction, rate, dt):
        nodes = np.r_[self.alpha, np.asarray(endpoints, dtype=float)]
        times = np.asarray(times, dtype=float)
        if nodes.shape != times.shape or len(nodes) < 2:
            raise ValueError('Continuous CL witness requires one node per boundary')
        limits = self.mission.data['limits']
        lower = max(float(self.db.A[0]), float(limits['alpha_rad'][0]))
        upper = min(float(self.db.A[-1]), float(limits['alpha_rad'][1]))
        control_valid = bool(np.isfinite(nodes).all() and np.all(nodes >= lower-1e-9)
                             and np.all(nodes <= upper+1e-9)
                             and np.all(np.diff(times) > 0)
                             and np.all(abs(np.diff(nodes)/np.diff(times))
                                        <= limits['alpha_rate_radps']+1e-9))
        rollout = self.plant.integrate(x, nodes, times, source, target, fraction, rate, dt)
        # Keep physical-grid validity and independent control validity separate.
        result = dict(rollout)
        result['valid'] = bool(rollout['valid'] and control_valid)
        return result


class ContinuousCLReachability:
    """Same outer-envelope/projection rules; new piecewise-linear witnesses."""

    def __init__(self, plant):
        self.plant = plant

    def check(self, x, alpha, transition, t, raw, enabled=False):
        screen = CLReachability(_WitnessPlant(self.plant, alpha))
        result = screen.check(x, alpha, transition, t, raw, enabled=enabled)
        result.update(
            reachability_method='path_limit_outer_box_and_audited_continuous_alpha_extremes',
            witness_control_interpolation='piecewise_linear_from_actual_alpha',
            outer_envelope_semantics='necessary_conservative_slot_bounds_not_complete_reachable_set')
        return result
