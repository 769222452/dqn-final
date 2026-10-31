"""Fixed-CL real dual-OCP experiment; refuses existing output directories."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from continuous_tracking.experiment import DEFAULT_CONFIG, run_constant


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--cl', type=float, default=.025)
    parser.add_argument('--max-decisions', type=int, default=400)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--legacy-reference',action='store_true',help='Run unchanged surrogate pipeline as a separate reference')
    args = parser.parse_args()
    if args.legacy_reference:
        from continuous_tracking.reference import run_reference
        result=run_reference(args.out,args.cl,args.max_decisions)
    else:
        result = run_constant(args.out, args.cl, args.max_decisions, args.config)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, ensure_ascii=False, indent=2))
