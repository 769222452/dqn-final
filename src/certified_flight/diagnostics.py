"""Constraint margin names and RK4-consistent interval statistics of plant rollouts."""
import numpy as np

MARGIN_NAMES = ('altitude_lower', 'altitude_upper', 'velocity_lower', 'velocity_upper',
                'gamma_lower', 'gamma_upper', 'dynamic_pressure', 'heat_flux', 'load_factor',
                'mach_lower', 'mach_upper')


def interval_moments(rollout, row=8, horizon=3., target=0.):
    """Mean and RMSE of one sample row with RK4 quadrature weights (the fifth end-state sample is excluded)."""
    if rollout is None:
        return {'mean': None, 'rmse': None}
    n = int(round(min(horizon, len(rollout['times'])*rollout['dt'])/rollout['dt']))
    if n <= 0:
        return {'mean': None, 'rmse': None}
    q = np.asarray(rollout['samples'])[row, :5*n].reshape(n, 5)[:, :4]
    weights = np.array([1., 2., 2., 1.])/6
    return {'mean': float(np.mean(q@weights)),
            'rmse': float(np.sqrt(np.mean(((q-target)**2)@weights)))}


def constraint_evidence(rollout, threshold=1e-3):
    """Names of violated (< -1e-7) and near-active (< threshold) margins."""
    if rollout is None:
        return [], []
    v = np.asarray(rollout['margins'])
    return ([name for name, value in zip(MARGIN_NAMES, v) if value < -1e-7],
            [name for name, value in zip(MARGIN_NAMES, v) if value < threshold])
