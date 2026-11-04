"""The project's frozen native-mesh audit, ported without file I/O (classification logic unchanged).

Source: upstream_matlab/source/tools/geometry_audit/scan.py `audit` and mesh_checks.py (rules:
upstream_matlab/source/results/v5_geometry_audit/classification_rules.json). The functions of mesh_checks are
imported from the archived copy, not re-implemented. Removed: the job/case folder handling, JSON dumps of
intersection records and the opposite-diagonal sensitivity run. Kept: numeric checks, per-skin mesh
diagnostics, upper/lower cross intersections, self intersections, the six x-sections, boundary checks and the
VALID / INVALID / AMBIGUOUS rule exactly as written there.

    python experiments/aero_fidelity/family/native_audit.py path/to/native_geometry.mat
"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'upstream_matlab/source/tools/geometry_audit'))

from mesh_checks import intersections, mesh_diagnostics, section_compare  # noqa: E402


def native_audit(mat):
    m = loadmat(mat)
    lower = np.stack([m['wav_'+a] for a in 'xyz'], -1)
    upper = np.stack([m['wav_up_'+a] for a in 'xyz'], -1)
    result = dict(numeric_checks={'nan_count': int(np.isnan(lower).sum()+np.isnan(upper).sum()),
                                  'inf_count': int(np.isinf(lower).sum()+np.isinf(upper).sum()),
                                  'complex_count': int(np.count_nonzero(np.imag(lower))+np.count_nonzero(np.imag(upper)))})
    if any(result['numeric_checks'].values()):
        result.update(classification='INVALID', failure_reason='Nonfinite/complex native coordinates')
        return result
    length = float(np.ptp(np.concatenate([lower.reshape(-1, 3), upper.reshape(-1, 3)]), axis=0).max())
    eps = max(128*np.finfo(float).eps*length, 1e-9*length)
    area_eps = 1e-14*length**2
    ld, lt, lf, li = mesh_diagnostics(lower, eps, area_eps)
    ud, ut, uf, ui = mesh_diagnostics(upper, eps, area_eps)
    le_error = float(np.linalg.norm(lower[:, 0]-upper[:, 0], axis=1).max())
    leading = np.stack([lower[:-1, 0], lower[1:, 0]], 1)
    cross = intersections(lt, ut, eps, area_eps, boundary=leading if le_error <= 4*eps else np.empty((0, 2, 3)))
    sl = intersections(lt, lt, eps, area_eps, self_mode=True)
    su = intersections(ut, ut, eps, area_eps, self_mode=True)
    points = np.concatenate([r['points_program'] for r in cross['records']]) if cross['records'] else None
    x0 = float(min(lower[:, :, 0].min(), upper[:, :, 0].min()))
    x1 = float(max(lower[:, :, 0].max(), upper[:, :, 0].max()))
    sections = [section_compare(lt, ut, x0+f*(x1-x0), eps) for f in [.1, .25, .5, .75, .9, 1.]]
    ok = [s for s in sections if s['status'] == 'OK']
    worst = min(ok, key=lambda s: s['min_thickness']) if ok else None
    boundary = {'leading_edge_max_gap': le_error,
                'symmetry_max_abs_y': float(max(np.abs(lower[-1, :, 1]).max(), np.abs(upper[-1, :, 1]).max())),
                'all_native_y_nonnegative': bool(min(lower[:, :, 1].min(), upper[:, :, 1].min()) >= -eps)}
    result.update(length_m=length, surface_intersection_count=cross['count'],
                  self_intersection_count=sl['count']+su['count'],
                  intersection_x_over_L=None if points is None else [float(points[:, 0].min()/length), float(points[:, 0].max()/length)],
                  unapproved_degenerate=ld['unapproved_degenerate_count']+ud['unapproved_degenerate_count'],
                  normal_issue_count=ld['normal_issue_count']+ud['normal_issue_count'],
                  single_valued_graph=bool(ld['single_valued_xy_graph_certificate'] and ud['single_valued_xy_graph_certificate']),
                  min_section_thickness=worst['min_thickness'] if worst else None,
                  tail_min_thickness=sections[5].get('min_thickness'), sections_resolved=len(ok),
                  boundary_checks=boundary)
    reasons, ambiguous = [], []
    if result['surface_intersection_count']:
        reasons.append('Actual non-leading-edge upper/lower triangle intersection')
    if result['self_intersection_count']:
        reasons.append('Actual surface self-intersection')
    if result['unapproved_degenerate']:
        reasons.append('Undocumented degenerate native faces')
    if not result['single_valued_graph']:
        ambiguous.append('Single-valued XY graph not certified')
    if result['normal_issue_count']:
        ambiguous.append('Adjacent native normals reverse')
    if le_error > 4*eps or boundary['symmetry_max_abs_y'] > 4*eps or not boundary['all_native_y_nonnegative']:
        ambiguous.append('Boundary connectivity unresolved')
    if len(ok) != 6:
        ambiguous.append('Some required sections unresolved')
    if worst and worst['min_thickness'] < -eps and not result['surface_intersection_count']:
        ambiguous.append('Negative thickness without triangle crossing confirmation')
    result['classification'] = 'INVALID' if reasons else ('AMBIGUOUS' if ambiguous else 'VALID')
    result['failure_reason'] = '; '.join(reasons+ambiguous) or 'All frozen native-mesh checks passed; OPEN_BASE declared'
    return result


if __name__ == '__main__':
    r = native_audit(sys.argv[1])
    print(json.dumps({k: v for k, v in r.items() if k != 'boundary_checks'}, indent=1, default=float))
