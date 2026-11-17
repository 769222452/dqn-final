"""P2 port, geometry generator and library metrics against the stored F8D2000 data."""
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from libdesign import geometry as Gm, metrics as Mx, p2 as P

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT/'data'/'F8D2000_source'/'samples'


@pytest.fixture(scope='module')
def f8d():
    z = np.load(ROOT/'data'/'F8D2000'/'p2_grid.npz')
    return z, [str(x) for x in z['geometry_ids']]


@pytest.mark.parametrize('gid', ['F8D_00001', 'F8D_01351'])
def test_p2_port_reproduces_library_tables(f8d, gid):
    z, ids = f8d
    lo, up = Gm.load_mat(SAMPLES/gid/'native_geometry.mat')
    t = P.grid_table(lo, up)
    i = ids.index(gid)
    assert t['valid'].all()
    assert np.max(np.abs(t['CL']-z['CL'][i])) < 1e-12
    assert np.max(np.abs(t['CD']-z['CD_pressure'][i])) < 1e-12
    assert np.max(np.abs(t['Cm']-z['Cm_pressure'][i])) < 1e-12


def test_p2_port_reproduces_g42_of_the_298_library():
    z = np.load(ROOT/'data'/'L298'/'p2_grid.npz')
    j = [str(x) for x in z['geometry_ids']].index('G42')
    t = P.grid_table(*Gm.parent('G42'))
    assert np.max(np.abs(t['CD']-z['CD_pressure'][j])) < 1e-12


@pytest.mark.parametrize('gid', ['F8D_00001', 'F8D_01351'])
def test_generator_reproduces_library_shapes(gid):
    m = sio.loadmat(SAMPLES/gid/'native_geometry.mat')
    c = m['shape_coefficients_m'].ravel()
    lo, up = Gm.deform(Gm.parent(str(m['parent_geometry_id'][0])), c[:4], c[4:])
    lo0, up0 = Gm.load_mat(SAMPLES/gid/'native_geometry.mat')
    assert np.max(np.abs(lo-lo0)) < 1e-12 and np.max(np.abs(up-up0)) < 1e-12


def test_leading_edge_is_fixed_by_every_basis():
    par = Gm.parent('F2K_01233')
    names = ('front', 'rear', 'tail', 'span', 'flap', 'twist')
    lo, up = Gm.deform(par, [.1]*6, [-.1]*6, names)
    assert np.array_equal(lo[:, 0], par[0][:, 0]) and np.array_equal(up[:, 0], par[1][:, 0])
    assert np.array_equal(lo[0], par[0][0]) and np.array_equal(up[0], par[1][0])
    assert np.array_equal(lo[..., :2], par[0][..., :2]) and np.array_equal(up[..., :2], par[1][..., :2])


def test_fast_checks_agree_with_reference_checks():
    par = Gm.parent('G42')
    F = Gm.Fixed(par)
    for cu, cl in (([0, 0, .04, 0], [0, 0, -.035, 0]), ([0, 0, .12, 0], [0, 0, -.1, 0]), ([.02, 0, 0, 0], [.05, 0, 0, 0])):
        lo, up = Gm.deform(par, cu, cl)
        assert F.check(lo, up) == Gm.check(lo, up)[0]
        assert F.volume(lo, up) == pytest.approx(Gm.volume(lo, up), rel=1e-9)


def test_metrics_reproduce_the_f8d2000_summary():
    m, _ = Mx.library_metrics(ROOT/'data'/'F8D2000'/'p2_grid.npz')
    assert m['static']['gain'] == pytest.approx(0.004109112132145154, rel=1e-3)   # alpha step 0.02 deg vs. the original root
    assert m['static']['n_winners'] == 6
