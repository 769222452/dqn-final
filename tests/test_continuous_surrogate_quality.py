"""Tracking quality is an explicit diagnostic, independent of physical safety."""
from copy import deepcopy

import numpy as np
import pytest

from continuous_surrogate.safety import ContinuousSafety


def judge(audit):
    safety = ContinuousSafety.__new__(ContinuousSafety)
    safety.quality_tolerance = .005
    safety.mean_tolerance = .0025
    safety.peak_tolerance = .01
    safety.rate = .0872665
    return safety._quality(audit)


def good_audit():
    return dict(valid=True, CL_target=0., maximum_alpha_rate_degps=0., grids={
        grid: dict(CL_mean_first3=.0025, CL_rmse_first3=.005, CL_peak_error_first3=.01)
        for grid in ('0.1', '0.5')})


def test_tracking_quality_threshold_boundary_is_inclusive():
    quality = judge(good_audit())
    assert quality['passed']
    assert quality['thresholds'] == dict(mean_abs_error_max=.0025, rmse_max=.005, peak_abs_error_max=.01)


@pytest.mark.parametrize('field', ['CL_mean_first3', 'CL_rmse_first3', 'CL_peak_error_first3'])
@pytest.mark.parametrize('grid', ['0.1', '0.5'])
def test_each_tracking_threshold_applies_on_each_audit_grid(field, grid):
    audit = good_audit()
    audit['grids'][grid][field] += .0001
    original = deepcopy(audit)
    quality = judge(audit)
    assert not quality['passed'] and quality['status'] == 'TRACKING_DEVIATION'
    assert audit == original and audit['valid']  # No physical rejection is fabricated.


def test_rate_limit_activity_does_not_replace_tracking_quality():
    audit = good_audit()
    audit['maximum_alpha_rate_degps'] = np.rad2deg(.0872665)
    quality = judge(audit)
    assert quality['passed'] and quality['rate_limit_active']
