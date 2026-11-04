"""Write an SU2 Euler config for the half model and run SU2 (OpenMP). usage: local_cfd_run.py <run_dir> <mach> <alpha> [iters] [threads]"""
import os, subprocess, sys, time
from pathlib import Path

run = Path(sys.argv[1]); M = sys.argv[2]; A = sys.argv[3]
iters = sys.argv[4] if len(sys.argv) > 4 else '4000'
thr = sys.argv[5] if len(sys.argv) > 5 else '10'
scheme = sys.argv[6] if len(sys.argv) > 6 else 'second'
restart = len(sys.argv) > 7 and sys.argv[7] == 'restart'
SREF_FULL = 6.931920786884801
cfg = f"""% V6 wedge/waverider inviscid (Euler) half-model run. Pressure-only forces; no viscosity, no base-flow model.
SOLVER= EULER
MATH_PROBLEM= DIRECT
RESTART_SOL= {'YES' if restart else 'NO'}
MACH_NUMBER= {M}
AOA= {A}
SIDESLIP_ANGLE= 0.0
FREESTREAM_OPTION= TEMPERATURE_FS
FREESTREAM_TEMPERATURE= 226.5
FREESTREAM_PRESSURE= 1197.0
GAMMA_VALUE= 1.4
GAS_CONSTANT= 287.058
REF_ORIGIN_MOMENT_X= 2.0
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= -0.9
REF_LENGTH= 5.0
REF_AREA= {SREF_FULL / 2:.12f}
REF_DIMENSIONALIZATION= FREESTREAM_PRESS_EQ_ONE
MARKER_EULER= ( BODY, BASE )
MARKER_SYM= ( SYMMETRY )
MARKER_FAR= ( FARFIELD )
MARKER_PLOTTING= ( BODY, BASE )
MARKER_MONITORING= ( BODY, BASE )
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= {os.environ.get('SU2_CFL', '1.0')}
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 1.05, 0.5, {os.environ.get('SU2_CFLMAX', '20.0')} )
ITER= {iters}
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 10
CONV_NUM_METHOD_FLOW= {os.environ.get('SU2_FLUX', 'ROE')}
ENTROPY_FIX_COEFF= 0.1
MUSCL_FLOW= {'NO' if scheme == 'first' else 'YES'}
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05
TIME_DISCRE_FLOW= EULER_IMPLICIT
CONV_RESIDUAL_MINVAL= -9
CONV_STARTITER= 10
CONV_FIELD= RMS_DENSITY
SCREEN_OUTPUT= INNER_ITER RMS_DENSITY RMS_ENERGY LIFT DRAG MOMENT_Y CFL_NUMBER
SCREEN_WRT_FREQ_INNER= 50
HISTORY_OUTPUT= ITER RMS_RES AERO_COEFF AERO_COEFF_SURF
OUTPUT_FILES= RESTART, CSV, SURFACE_CSV
OUTPUT_WRT_FREQ= 500
MESH_FILENAME= {os.environ.get('SU2_MESH', 'mesh.su2')}
MESH_FORMAT= SU2
CONV_FILENAME= history
SOLUTION_FILENAME= solution_flow
RESTART_FILENAME= restart_flow
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
"""
(run / 'config.cfg').write_text(cfg)
exe = r'D:\v6-mission-morph-core\tools_cfd\SU2x\bin\SU2_CFD.exe'
t = time.time()
with open(run / 'su2.log', 'w') as lf:
    p = subprocess.run([exe, '-t', thr, 'config.cfg'], cwd=run, stdout=lf, stderr=subprocess.STDOUT)
print('SU2 exit', p.returncode, 'wall %.1f s' % (time.time() - t))
