"""Combustion model helper functions"""

import numpy as np


def eval_comb_weighting_fun(tau: float, tau_min: float, tau_max: float,
                            weighting_fun: str = 'smooth') -> float:
    """Evaluate combustion weighting function
    
    This function determines the weighting between premixed and diffusive
    combustion modes based on ignition delay.
    
    Args:
        tau: Ignition delay [s]
        tau_min: Minimum ignition delay for transition [s]
        tau_max: Maximum ignition delay for transition [s]
        weighting_fun: Type of weighting function ('smooth' or 'non-smooth')
    
    Returns:
        gamma: Weighting factor [0-1], where 0 is pure diffusive and 1 is pure premixed
    """
    if weighting_fun == 'non-smooth':
        if tau < tau_min:
            gamma = 0
        elif tau > tau_max:
            gamma = 1
        else:
            delta_tau = tau_max - tau_min
            gamma = 1 / delta_tau * tau - tau_min / delta_tau
    else:  # smooth
        slope = 1 / (tau_max - tau_min)
        shift = (tau_max - tau_min) / 2 + tau_min
        gamma = 1 / (1 + np.exp(-4 * slope * (tau - shift)))
    
    return gamma
