"""Create matlab_diag/build_geometry_core_diag.m from matlab/active/build_geometry_core.m by two
exact textual patches that only ADD outputs (no numerical statement is changed), and write the diff.

Patches:
  1. function name/signature: extra output `flowdiag`;
  2. before the final `diagnostics=struct(...)`: pack the already-computed design flow quantities
     (p_wav, velocity, shock angles, wall-streamline states, free-stream state) into `flowdiag`.
"""
from __future__ import annotations

import difflib

import common as C

SRC = C.ACTIVE / "build_geometry_core.m"
DST = C.HERE / "matlab_diag/build_geometry_core_diag.m"

P1 = ("function [wav_x,wav_y,wav_z,wav_up_x,wav_up_y,wav_up_z,diagnostics]=build_geometry_core(",
      "function [wav_x,wav_y,wav_z,wav_up_x,wav_up_y,wav_up_z,diagnostics,flowdiag]=build_geometry_core_diag(")
P2_ANCHOR = "    diagnostics=struct('native_M',M,"
P2_ADD = """    % ---- V6 aero-physics-validation diagnostic outputs (added; no computation above is changed) ----
    flowdiag=struct();
    flowdiag.p_wav=p_wav; flowdiag.vx_wav=vx_wav; flowdiag.vy_wav=vy_wav; flowdiag.vz_wav=vz_wav; flowdiag.p_wav_up=p_wav_up;
    flowdiag.shock_angle=shock_angle; flowdiag.U_index=U_index; flowdiag.V_index=V_index; flowdiag.index=index;
    flowdiag.x_y_z_shock=x_y_z_shock; flowdiag.h_center_body=h_center_body;
    flowdiag.wall_x=squeeze(x_inter_3(:,1,:)); flowdiag.wall_y=squeeze(y_inter_3(:,1,:)); flowdiag.wall_z=squeeze(z_inter_3(:,1,:));
    flowdiag.wall_p=squeeze(p_l_3(:,1,:)); flowdiag.wall_r=squeeze(r_l_3(:,1,:)); flowdiag.wall_v=squeeze(v_l_3(:,1,:));
    flowdiag.wall_a=squeeze(a_l_3(:,1,:));
    flowdiag.p0=p0; flowdiag.r0=r0; flowdiag.m0=m0; flowdiag.v0=v0; flowdiag.Ma=Ma; flowdiag.AOA=AOA; flowdiag.rat=rat; flowdiag.Vector_V=Vector_V;
    flowdiag.x_fct=x_fct; flowdiag.y_fct=y_fct; flowdiag.z_fct=z_fct; flowdiag.u_fct=u_fct; flowdiag.v_fct=v_fct;
    flowdiag.x_fct_1=x_fct_1; flowdiag.y_fct_1=y_fct_1; flowdiag.z_fct_1=z_fct_1;
"""


def main():
    src = SRC.read_text(encoding="utf-8")
    assert src.count(P1[0]) == 1 and src.count(P2_ANCHOR) == 1
    out = src.replace(P1[0], P1[1]).replace(P2_ANCHOR, P2_ADD + P2_ANCHOR)
    DST.write_text(out, encoding="utf-8")
    diff = "".join(difflib.unified_diff(src.splitlines(True), out.splitlines(True),
                                        "matlab/active/build_geometry_core.m",
                                        "tools/v6_aero_physics_validation/matlab_diag/build_geometry_core_diag.m"))
    (C.HERE / "matlab_diag/build_geometry_core_diag.diff").write_text(diff, encoding="utf-8")
    C.dump(C.HERE / "matlab_diag/PROVENANCE.json",
           {"source": "matlab/active/build_geometry_core.m", "source_sha256": C.sha_file(SRC),
            "diag_sha256": C.sha_file(DST), "active_copy": "matlab_diag/active_copy (verbatim copy of matlab/active/*.m)",
            "active_copy_sha256": {p.name: C.sha_file(p) for p in sorted((C.HERE / "matlab_diag/active_copy").glob("*.m"))},
            "active_sha256": {p.name: C.sha_file(p) for p in sorted(C.ACTIVE.glob("*.m"))},
            "changes": ["renamed function + extra output 'flowdiag'", "flowdiag struct packed before diagnostics=struct(...)"],
            "created_utc": C.utc()})
    print(diff)


if __name__ == "__main__":
    main()


