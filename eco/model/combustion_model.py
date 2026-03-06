"""Combustion model"""

import numpy as np
from .subfunctions.combustion import eval_comb_weighting_fun


def combustion_model(q_comb: float, m_fuel_prep: float, tau_ign: float,
                    we: float, par) -> float:
    """Combustion model based on Chmela approach
    
    This model calculates heat release rate based on prepared fuel mass,
    ignition delay, and engine speed. It uses a mixed combustion approach
    between premixed and diffusive combustion.
    
    Args:
        q_comb: Total heat released by burned fuel so far [J]
        m_fuel_prep: Total injected fuel ready to be burned (at t-tau_ign) [kg]
        tau_ign: Ignition delay [s]
        we: Engine speed [1/s]
        par: Parameter set
    
    Returns:
        dq_comb_dphi: Heat release rate [J/degCA]
    """
    # Time passed per °CA and vice versa
    dphi_dt = we * 360
    dt_dphi = 1 / dphi_dt
    
    # Available fuel energy potential in [J]
    q_dis = 0  # General assumption
    dq_dis_dphi = 0
    q_fuel_pot = (m_fuel_prep * par.thermo['fuel']['low_heat_val'] * 
                  par.comb['eta_comb'] - (q_comb - q_dis))
    
    # Weighting factor for mixing rate in [1]
    # gamma = 0    : pure diffusive mixing rate
    # gamma = 1    : pure premixed mixing rate
    # 0 < gamma < 1: mixed mixing rate
    gamma = eval_comb_weighting_fun(tau_ign,
                                    par.comb['gamma']['tau_min'],
                                    par.comb['gamma']['tau_max'])
    
    # Mixing rate model parameters
    a_dif = par.comb['c_dif']['a']
    b_dif = par.comb['c_dif']['b']
    a_pre = par.comb['c_pre']['a']
    b_pre = par.comb['c_pre']['b']
    
    # Heat release rate in [J/degCA]
    de_nom = (1 - q_fuel_pot * (gamma * a_pre + (1 - gamma) * a_dif))
    
    dq_comb_dphi = ((q_fuel_pot * dt_dphi * (gamma * b_pre +
                    (1 - gamma) * b_dif) + dq_dis_dphi) /
                   de_nom)
    
    # Limit dQCombdPhi to 0
    eps = 1e-4
    dq_comb_dphi = 0.5 * dq_comb_dphi + 0.5 * np.sqrt(eps**2 + dq_comb_dphi**2)
    
    return dq_comb_dphi
