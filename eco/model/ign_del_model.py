"""Ignition delay model"""

import numpy as np
from .ignition_delay_joerg import ignition_delay_joerg
from .subfunctions.ignition_delay import saturate_input


def ign_del_model(p_cyl: float, theta_cyl: float, xi_o2: float, 
                 par_model) -> float:
    """Calculate total ignition delay
    
    Args:
        p_cyl: Actual cylinder pressure [Pa]
        theta_cyl: Actual cylinder temperature [K]
        xi_o2: Oxygen mass fraction [-]
        par_model: Parameter structure
    
    Returns:
        tau_ign: Ignition delay [s]
    """
    # Chemical ignition delay time in [s]
    if par_model.opts['ignition_delay_model'] == 'Joerg' or par_model.opts['ignition_delay_model'] == 1:
        tau_ign_chem = ignition_delay_joerg(p_cyl, theta_cyl, xi_o2) * 1e-3
    else:
        raise ValueError('Wrong specification for ignition delay model.')
    
    # Physical ignition delay time in [s]
    tau_ign_phys = par_model.ign['phys']['const']
    
    # Constant bounds
    tau_ign_min = par_model.ign['tau_min']
    tau_ign_max = par_model.ign['tau_max']
    
    # Total ignition delay time in [s]
    tau_ign = saturate_input(tau_ign_chem + tau_ign_phys, tau_ign_min, tau_ign_max)
    
    return tau_ign
