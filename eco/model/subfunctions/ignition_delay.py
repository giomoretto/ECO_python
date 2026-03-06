"""Ignition delay model helper functions"""

import numpy as np


def saturate_input(u: np.ndarray, u_min: float, u_max: float,
                  saturation_fun: str = 'smooth') -> np.ndarray:
    """Saturate input to specified bounds
    
    Args:
        u: Input value(s)
        u_min: Minimum bound
        u_max: Maximum bound
        saturation_fun: Type of saturation ('smooth' or 'exact')
    
    Returns:
        y: Saturated output
    """
    u = np.atleast_1d(u)
    y = u.copy()
    
    if saturation_fun == 'exact':
        y = np.clip(y, u_min, u_max)
    else:  # smooth
        epsilon = 1e-6
        h = (u_max - u_min) / 2
        y_mean = (u_min + u_max) / 2
        
        y = (h / 2 * (np.sqrt(epsilon + ((u - y_mean) / h + 1) ** 2) -
                     np.sqrt(epsilon + ((u - y_mean) / h - 1) ** 2)) + y_mean)
    
    # Return scalar if input was scalar
    if y.size == 1:
        return float(y[0])
    return y
