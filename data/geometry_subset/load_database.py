"""Read the portable snapshot. Python + NumPy; no campaign imports or writes."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

def load_database(directory=None, verify=False):
    root = Path(directory).resolve() if directory else Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'VERIFIED':
        raise ValueError('Database export is not verified')
    if verify:
        for item in manifest['files']:
            file = (root / item['path']).resolve()
            if not file.is_relative_to(root):
                raise ValueError('Invalid relative path')
            if hashlib.sha256(file.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('File hash mismatch: ' + item['path'])
    with np.load(root / 'aerodynamics/p2_grid.npz', allow_pickle=False) as stored:
        data = {key: stored[key].copy() for key in stored.files}
    if list(data['CL'].shape) != manifest['tensor_shape']:
        raise ValueError('Tensor shape mismatch')
    if data['geometry_ids'].tolist() != manifest['geometry_ids']:
        raise ValueError('Action-index mapping mismatch')
    return {'root': root, 'manifest': manifest, **data}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    db = load_database(verify=args.verify)
    print(json.dumps({'status':'PASS','geometry_count':len(db['geometry_ids']),
        'shape':list(db['CL'].shape),'usable_p2_points':int(db['usable_p2'].sum()),
        'all_manifest_files_verified':args.verify}, indent=2))
