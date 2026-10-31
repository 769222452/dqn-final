"""Experimental continuous-alpha, full-horizon CL-tracking control.

This protocol is independent of the trained surrogate and legacy certificates.
It models a rate-limited ideal inner loop, not identified attitude dynamics.
"""

PROTOCOL = 'continuous-alpha-full-horizon-tracking/1'
