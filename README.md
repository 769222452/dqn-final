# Evidence archive: audits, checks and experiment results

Created on 2026-11-04 from the evidence branches of the repository. Only the evidence directories are in the tree; the full source of
each experiment is reachable through the merge parents (each source branch tip is a parent of one commit of this branch), so no history is lost.

| directory | what it holds | source branch (tip) |
|---|---|---|
| `cloud_results/v14_final/` | original V14 and V16 cloud results (archive, manifest, summaries) | cloud-results-v14-final-20261031 (456ceed) |
| `cloud_provenance/`, `CLOUD_SOURCE_ZH.md` | provenance and checksums of the original V16 cloud source | cloud-v16-original-20261031 (71eb48e) |
| `evidence/aero_fidelity/20261102_4d211b1/` | controlled family version 1: generation and aborted geometry check | aero-family-results-20261102 (2acaaa5) |
| `evidence/aero_fidelity/20261102_family_v2/` | family version 2: 180 generator calls, frozen check (no admissible family) | aero-family-v2-results-20261102 (6d41c60) |
| `evidence/aero_fidelity/20261102_family_v3/` | family version 3: 189 jobs, frozen selection evidence | aero-family-v3-results-20261102 (b8b04ef) |
| `evidence/aero_fidelity/20261102_family_v4/` | family version 4: selected family and the 29-case SU2 Euler run (total forces) | aero-family-v4-results-20261102 (735c550) |
| `evidence/aero_fidelity/20261102_cfd_v5/` | CFD version 5: skin-only forces, diagnosis of the base-cap artefact | aero-family-v5-results-20261102 (bb28355) |
| `evidence/aero_fidelity/20261103_optimal_family/` | optimal-family test (d1 fixed), wedge + friction and Euler confirmation | aero-opt-results-20261103 (9b3e613) |
| `evidence/network_architecture/20261104/` | advantage-network architecture study (17 structures, forward-chaining) | network-architecture-results-20261104 (fc5f56d) |

How the experiments are interpreted (declared criteria, amendments, outcomes) is written in `experiments/aero_fidelity/README.md` of the
`project-latest` branch, sections 3 to 3i. Every evidence directory carries its own README, SHA256SUMS and provenance files.
