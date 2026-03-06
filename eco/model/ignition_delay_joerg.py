"""Ignition delay Joerg model"""

import numpy as np


def ignition_delay_joerg(p: float, T: float, x_o2: float, 
                         fuel_type: str = 'Diesel') -> float:
    """Calculate ignition delay using Joerg model
    
    Args:
        p: Cylinder pressure [Pa]
        T: Cylinder temperature [K]
        x_o2: Mass fraction of oxygen [0-1]
        fuel_type: Type of fuel ('Diesel' or 'nHeptane')
    
    Returns:
        tau_chem: Chemical ignition delay [ms]
    """
    p_ref = 40e5
    x_o2_ref = 0.224
    R = 1
    
    def func_arr(p, T, x_o2, c_pres, c_o2, k, ea):
        """Arrhenius function"""
        return (k * (p / p_ref)**c_pres * (x_o2 / x_o2_ref)**c_o2 * 
                np.exp(ea / T / R))
    
    if fuel_type == 'Diesel':
        tau_ht = func_arr(p, T, x_o2, -1.05, -1.5, 1.25e-07, 15813)
        tau_lt = func_arr(p, T, x_o2, 0, 0, 1.09e-08, 13188)
        
        a_ht = func_arr(p, T, x_o2, -1, -1, 3.11e-03, 0)
        a_lt = func_arr(p, T, x_o2, 0, 0, 4.51e-12, 21079)
    
    elif fuel_type == 'nHeptane':
        tau_ht = func_arr(p, T, x_o2, -1.1, -1.5, 1.3e-07, 15813)
        tau_lt = func_arr(p, T, x_o2, 0, -0.2, 2.65e-08, 13188)
        
        a_ht = func_arr(p, T, x_o2, -0.5, -0.7, 3.39e-3, 0)
        a_lt = func_arr(p, T, x_o2, 0, 1, 4.51e-12, 21079)
    else:
        raise ValueError(f"Unknown fuel type: {fuel_type}")
    
    w_ht = a_ht / (a_ht + a_lt)
    w_lt = 1 - w_ht
    
    tau_chem = w_ht * tau_ht + w_lt * tau_lt
    
    return tau_chem
