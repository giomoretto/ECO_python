"""Ignition delay model helper functions (CasADi version)"""

import casadi as ca


def saturate_input(u, u_min, u_max):
    """Smooth saturation compatible with CasADi symbolics.

    Args:
        u: Input value (CasADi symbolic or numeric)
        u_min: Minimum bound
        u_max: Maximum bound

    Returns:
        y: Saturated output
    """
    epsilon = 1e-6
    h = (u_max - u_min) / 2
    y_mean = (u_min + u_max) / 2

    y = (h / 2 * (ca.sqrt(epsilon + ((u - y_mean) / h + 1) ** 2) -
                  ca.sqrt(epsilon + ((u - y_mean) / h - 1) ** 2)) + y_mean)
    return y
