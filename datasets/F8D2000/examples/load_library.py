"""Load a full-precision geometry and its [Mach, alpha] pressure-aero tables."""
from pathlib import Path
import argparse
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]


def load_geometry_and_aero(geometry_id, root=ROOT):
    root = Path(root)
    with np.load(root / 'aerodynamics/p2_grid.npz', allow_pickle=False) as z:
        ids = z['geometry_ids'].tolist()
        if geometry_id not in ids:
            raise ValueError(f'Unknown candidate geometry: {geometry_id}')
        index = ids.index(geometry_id)
        aero = {k:z[k][index].copy() for k in
                ['CL','CD_pressure','Cm_pressure','model_valid','usable_p2']}
        aero.update(mach_grid=z['mach_grid'].copy(),alpha_deg_grid=z['alpha_deg_grid'].copy())
    geometry = loadmat(root / f'geometries/{geometry_id}/native_geometry.mat', simplify_cells=True)
    return {'geometry_id':geometry_id,'action_index':index,'geometry':geometry,**aero}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('geometry_id', nargs='?', default='F8D_01351')
    args = parser.parse_args()
    data = load_geometry_and_aero(args.geometry_id)
    print(f"{data['geometry_id']}: action_index={data['action_index']}; "
          f"surface={data['geometry']['wav_z'].shape}; aero={data['CL'].shape}; "
          f"usable={int(data['usable_p2'].sum())}")
