"""Combustion model helper functions (CasADi version)"""

import casadi as ca


def eval_comb_weighting_fun(tau, tau_min, tau_max):
    """Evaluate combustion weighting function (smooth sigmoid, CasADi).

    Determines the weighting between premixed and diffusive combustion modes
    based on ignition delay.

    Args:
        tau: Ignition delay [s]
        tau_min: Minimum ignition delay for transition [s]
        tau_max: Maximum ignition delay for transition [s]

    Returns:
        gamma: Weighting factor [0-1], 0 = pure diffusive, 1 = pure premixed
    """
    slope = 1 / (tau_max - tau_min)
    shift = (tau_max - tau_min) / 2 + tau_min
    gamma = 1 / (1 + ca.exp(-4 * slope * (tau - shift)))
    return gamma
