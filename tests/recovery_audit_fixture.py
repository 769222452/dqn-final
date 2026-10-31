"""Complete deterministic dual-grid evidence for runner transaction tests."""
import numpy as np


def current_audit(context, nodes, times, cl, *, displacement=(0., 0., 0., 6000.),
                  actual_cl=None, valid=True, margin_name='dynamic_pressure'):
    """Mirror the plant audit schema without solving the flight dynamics."""
    initial = np.asarray(context['x'], float)
    delta = np.asarray(displacement, float)
    nodes, times = np.asarray(nodes, float), np.asarray(times, float)
    actual_cl = cl if actual_cl is None else actual_cl
    arrays, grids = {}, {}
    for key in ('0.1', '0.5'):
        dt = float(key)
        n = round(3./dt)
        tt = np.arange(n)*dt
        # The five sample columns represent entry, both RK midpoints, the
        # provisional endpoint and the accepted endpoint for each interval.
        fractions = ((tt[:, None]+dt*np.asarray([0., .5, .5, 1., 1.]))/3.).ravel()
        samples = np.zeros((12, 5*n))
        samples[:4] = (initial+fractions[:, None]*delta).T
        x = samples[:4, 4::5].T.copy()
        samples[4], samples[5], samples[6], samples[8], samples[10] = 100., 1000., 1., actual_cl, 10.
        index = np.clip(np.searchsorted(times, tt+1e-9, side='right')-1, 0, len(nodes)-2)
        arrays[key] = dict(x=x, samples=samples, times=tt, dt=dt,
            alpha=np.interp(tt, times, nodes), alpha_end=np.interp(tt+dt, times, nodes),
            alpha_rate=(np.diff(nodes)/np.diff(times))[index], valid=valid,
            min_margin=.5, margins=[.5])
        grids[key] = dict(valid=valid, min_margin=.5, margins={margin_name: .5},
            CL_mean_first3=actual_cl, CL_rmse_first3=.001,
            CL_peak_error_first3=abs(actual_cl-cl), end_state=x[-1].tolist())
    return dict(valid=valid, continuous_at_entry=True, control_valid=True, grids=grids,
        maximum_alpha_rate_degps=float(np.rad2deg(np.max(np.abs(np.diff(nodes)/np.diff(times)))))), arrays
