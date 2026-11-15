"""Heat-load budget screening (pressure-only aerodynamics, nominal state).

Budgets are fractions of the integrated stagnation heat load of the best unconstrained simple law (pressure.json).
For each budget: the best simple law of pressure.json that meets it, and the relaxed physical bound with the same
heat-load constraint (certified_flight.bound.relaxed_bound(heat_budget=...)), warm-started from the
unconstrained bound (experiments/improve/bounds/dh0_dv0.json).

    python experiments/perturb/screening/heat_budget.py --fraction 0.8 --out experiments/perturb/screening/heat_0.8.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
HERE = Path(__file__).resolve().parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fraction', type=float, required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    from certified_flight import bound as B
    from certified_flight.aero import AeroDatabase
    from certified_flight.build import runtime_config
    from certified_flight.mission import Mission
    sc = json.loads((HERE/'pressure.json').read_text())
    rows = [r for k in ('const', 'linear', 'two-phase') for r in sc[k]['grid'] if r['feasible']]
    ref = sc['const']['best']
    budget = a.fraction*ref['heat_load']
    ok = [r for r in rows if r['heat_load'] <= budget]
    law = max(ok, key=lambda r: r['range_km']) if ok else None
    physical = runtime_config(ROOT)['physical']
    mission = Mission(ROOT, physical['mission'])
    db = AeroDatabase(ROOT/physical['snapshot'])
    x0 = np.array(mission.data['initial_state']['x_h_V_gamma_s'], float)
    warm_sol = json.loads((ROOT/'experiments/improve/bounds/dh0_dv0.json').read_text())['relaxed']
    warm = dict(t=warm_sol['t'], x=warm_sol['x'], t_ctrl=warm_sol['t'][:-1], CL=warm_sol['CL'])
    sol = B.relaxed_bound(mission, db, x0, warm=warm, heat_budget=budget)
    out = dict(fraction=a.fraction, budget_J_m2=budget, reference_law=ref, best_law=law,
               bound=dict(status=sol['status'], range_km=sol['range_km'], verification=sol['verification'],
                          t=sol['t'], x=sol['x'], CL=sol['CL']))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(fraction=a.fraction, budget=budget, bound_km=sol['range_km'], status=sol['status'],
                          law_km=None if law is None else law['range_km'],
                          law={k: law[k] for k in ('kind', 'r', 'r0', 'r1', 'T', 'kp', 'kg') if law and k in law})))


if __name__ == '__main__':
    main()
