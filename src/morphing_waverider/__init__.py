"""morphing_waverider: V0 modular research framework.

Layers (each only depends on the layers above it, never on RL):
    utils  -> dynamics (atmosphere, 3-DOF) -> aero -> generator -> validation
           -> trajectory (CasADi OCP) -> envs (Gymnasium) -> rl (SB3 baselines)
"""

__version__ = "0.0.1"
