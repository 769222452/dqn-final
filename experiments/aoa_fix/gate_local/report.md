# AoA-sawtooth experiment report

## Gate (nominal state)

| policy | objective | completed | range_km | reversal_rate | intra_step_ptp_median_deg | boundary_jump_median_deg | sawtooth_opposition | power_3s_fraction | hf_rms_deg | CL_mean_abs_error_median | lookahead_offset_steps | fallback_steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| const0.025 | consistent | True | 2475.940 | 0.196 | 0.002 | 0.001 | 0.577 | 0.000 | 0.074 | 0.003 | 16 | 2 |
| const0.025 | v16 | True | 2456.448 | 0.329 | 1.404 | 1.406 | 0.985 | 0.510 | 0.503 | 0.004 | 19 | 2 |
| seed101 | consistent | True | 2477.807 | 0.142 | 0.002 | 0.001 | 0.410 | 0.000 | 0.129 | 0.003 | 6 | 4 |
| seed101 | v16 | True | 2460.650 | 0.332 | 1.407 | 1.401 | 0.995 | 0.684 | 0.501 | 0.004 | 4 | 0 |

- **const0.025**: G1_sawtooth_removed=True, G2_feasible=True, G3_range_kept=False, range_delta_km=19.492, reversal_rate_below_5pct=False, PASS=False, G3_no_loss=True, PROCEED_TO_HELDOUT=True
- **seed101**: G1_sawtooth_removed=True, G2_feasible=True, G3_range_kept=False, range_delta_km=17.157, reversal_rate_below_5pct=False, PASS=False, G3_no_loss=True, PROCEED_TO_HELDOUT=True

