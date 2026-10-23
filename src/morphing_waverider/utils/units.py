"""Unit conversion helpers. Internal representation is strictly SI (m, kg, s, rad)."""
import math

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
KM2M = 1000.0
M2KM = 1e-3
G0 = 9.80665  # m/s^2, standard gravity (used for load-factor normalization)


def deg(x_rad: float) -> float:
    return x_rad * RAD2DEG


def rad(x_deg: float) -> float:
    return x_deg * DEG2RAD


def km(x_m: float) -> float:
    return x_m * M2KM
