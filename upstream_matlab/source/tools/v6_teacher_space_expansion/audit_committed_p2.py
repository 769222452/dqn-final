"""Read-only identity and consistency audit of atomically committed campaign P2 files."""
from __future__ import annotations

from collections import defaultdict
from exp_common import *


def main():
    registry = pool().set_index('geometry_id')
    paths = sorted((OUT / 'aero').glob('*/*.csv')) + sorted((OUT / 'cases').glob('*/frozen49.csv'))
    paths = [p for p in paths if not p.name.endswith(('.queries.csv', '.tmp.csv'))]
    issues, files, pieces = [], [], []
    coordinate_hashes = defaultdict(set)
    mat_ids = set()
    mat_files = []
    rows = valid_rows = 0
    for path in paths:
        digest = sha(path)
        d = pd.read_csv(path, float_precision='round_trip')
        query = path.with_suffix('.queries.csv')
        if not query.exists():
            issues.append({'path':str(path.relative_to(OUT)), 'issue':'Missing actual query CSV'})
            continue
        q = pd.read_csv(query, float_precision='round_trip').rename(columns={'Mach':'M_flight'})
        keys = ['geometry_id', 'M_flight', 'alpha_deg']
        left = d[keys].sort_values(keys).reset_index(drop=True)
        right = q[keys].sort_values(keys).reset_index(drop=True)
        if len(left) != len(right) or not np.array_equal(left.to_numpy(), right.to_numpy()):
            issues.append({'path':str(path.relative_to(OUT)), 'issue':'Output/query key mismatch'})
        if d.duplicated(keys).any():
            issues.append({'path':str(path.relative_to(OUT)), 'issue':'Duplicate keys within committed output'})
        for gid, g in q.groupby('geometry_id'):
            if gid not in registry.index:
                issues.append({'geometry':gid, 'issue':'Missing geometry registry entry'})
                continue
            expected = (ROOT / registry.loc[gid, 'mat_path']).resolve()
            if any(Path(x).resolve() != expected for x in g.mat_path):
                issues.append({'geometry':gid, 'issue':'Query MAT path does not match registry'})
            mat_ids.add(gid)
        for gid, g in d.groupby('geometry_id'):
            coordinate_hashes[gid].update(g.v7_coordinate_hash.dropna().astype(str))
        physical = np.isfinite(d.CL) & np.isfinite(d.CD_pressure) & d.CD_pressure.gt(0)
        valid = true(d.model_valid)
        if (valid & ~physical).any():
            issues.append({'path':str(path.relative_to(OUT)), 'issue':'model_valid with nonphysical coefficients'})
        if sha(path) != digest:
            issues.append({'path':str(path.relative_to(OUT)), 'issue':'Committed output changed during audit'})
        rows += len(d)
        valid_rows += int((valid & physical).sum())
        pieces.append(d)
        files.append({'path':str(path.relative_to(OUT)), 'rows':len(d), 'sha256':digest,
                      'query_sha256':sha(query)})
    for gid in sorted(mat_ids):
        r = registry.loc[gid]
        mat = ROOT / r.mat_path
        digest = sha(mat)
        # The original pool stores file SHA256. New immutable records instead
        # store a coordinate hash; a missing optional byte hash is not a mismatch.
        if pd.notna(r.mat_sha256):
            kind = 'saved_file_sha256'
            if digest != r.mat_sha256:
                issues.append({'geometry':gid, 'issue':'MAT content hash changed from geometry registry'})
        else:
            kind = 'saved_native_coordinate_hash'
            if coordinate_hash(mat) != r.coordinate_hash:
                issues.append({'geometry':gid, 'issue':'Native coordinate hash changed from geometry registry'})
        mat_files.append({'geometry_id':gid, 'sha256':digest, 'identity_check':kind})
        if len(coordinate_hashes[gid]) != 1:
            issues.append({'geometry':gid, 'issue':'P2 coordinate fingerprint absent or inconsistent across outputs',
                           'fingerprints':sorted(coordinate_hashes[gid])})
    data = pd.concat(pieces, ignore_index=True)
    duplicate = data[data.duplicated(['geometry_id', 'M_flight', 'alpha_deg'], keep=False)]
    spread = duplicate.groupby(['geometry_id', 'M_flight', 'alpha_deg'])[['CL', 'CD_pressure']].agg(lambda s:s.max()-s.min())
    max_difference = float(spread.max().max()) if len(spread) else 0.0
    if max_difference > 1e-12:
        issues.append({'issue':'Overlapping P2 coefficient disagreement above 1e-12', 'max_difference':max_difference})
    result = {'utc':utc(), 'status':'PASS' if not issues else 'FAIL',
              'scope':'Snapshot of committed campaign outputs, actual MATLAB query CSVs and registered MAT content; later live commits are outside this snapshot',
              'output_files_checked':len(files), 'rows_checked':rows, 'usable_P2_rows':valid_rows,
              'registered_MAT_files_checked':len(mat_ids),
              'overlapping_rows':len(duplicate), 'max_overlap_coefficient_difference':max_difference,
              'issues':issues, 'files':files, 'mat_files':mat_files}
    dump(OUT / 'validation_committed_P2_identity_latest.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','mat_files')}, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
