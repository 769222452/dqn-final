"""v4c: the F8D2000B shapes the geometry generator can select (written next to the snapshot, read by AeroDatabase).

    python scripts/v4c_generator_candidates.py      # writes data/geometry_f8d2000b/generator_candidates.json

On a dense grid of the table support (96 Mach values x 57 CL values from -0.01 to 0.13) and for every requested drag level
(minimum drag, 0.2, 0.4, 0.6, 0.8, 1.0), the selected shape, the minimum- and maximum-drag shapes (they define the level
scale) and every shape within the generator's keep tolerance of the best level are collected, plus the start geometry.
The generator then chooses among these candidates only, and the reachability envelope ('any target may be chosen next')
covers exactly them. With all 2091 shapes the look-ahead guard spent ~80 % of its time evaluating shapes that are never
selected.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from certified_flight.aero import AeroDatabase  # noqa: E402

OUT = ROOT/'data'/'geometry_f8d2000b'/'generator_candidates.json'
START = 'F8D_01351'


def main():
    if OUT.exists():
        OUT.unlink()
    db = AeroDatabase(ROOT/'data'/'geometry_f8d2000b')
    assert db.candidates is None
    keep = AeroDatabase.LEVEL_KEEP
    U = {db.index[START]}
    for M in np.linspace(db.M[0], db.M[-1], 96):
        for cl in np.linspace(-.01, .13, 57):
            _, cd, ok = db.equal_lift(float(M), float(cl))
            if not ok.any():
                continue
            lo, hi = cd[ok].min(), cd[ok].max()
            U.add(int(np.argmin(np.where(ok, cd, np.inf))))
            U.add(int(np.argmax(np.where(ok, cd, -np.inf))))
            lv = np.where(ok, (cd-lo)/max(hi-lo, 1e-15), np.inf)
            for w in (.2, .4, .6, .8, 1.):
                d = np.abs(lv-w)
                U.update(int(k) for k in np.flatnonzero(d <= d.min()+keep))
    ids = sorted(db.ids[k] for k in U)
    OUT.write_text(json.dumps(dict(description=__doc__.strip().splitlines()[0], geometry_ids=ids,
                                   brake_shapes=sum(not g.startswith('F8D') for g in ids)), indent=1)+'\n', encoding='utf-8')
    print(f'{len(ids)} candidates ({sum(not g.startswith("F8D") for g in ids)} brake shapes) -> {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
