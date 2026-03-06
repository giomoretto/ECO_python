"""Combustion model (CasADi version)"""

import casadi as ca
from .subfunctions.combustion import eval_comb_weighting_fun


def combustion_model(q_comb, m_fuel_prep, tau_ign, we, par):
    """Combustion model based on Chmela approach (CasADi symbolic).

    Calculates heat release rate based on prepared fuel mass, ignition delay,
    and engine speed.

    Args:
        q_comb: Total heat released by burned fuel so far [J]
        m_fuel_prep: Total injected fuel ready to be burned (at t-tau_ign) [kg]
        tau_ign: Ignition delay [s]
        we: Engine speed [1/s]
        par: Parameter set

    Returns:
        dq_comb_dphi: Heat release rate [J/degCA]
    """
    dphi_dt = we * 360
    dt_dphi = 1 / dphi_dt

    q_fuel_pot = (m_fuel_prep * par.thermo['fuel']['low_heat_val'] *
                  par.comb['eta_comb'] - q_comb)

    gamma = eval_comb_weighting_fun(tau_ign,
                                    par.comb['gamma']['tau_min'],
                                    par.comb['gamma']['tau_max'])

    a_dif = par.comb['c_dif']['a']
    b_dif = par.comb['c_dif']['b']
    a_pre = par.comb['c_pre']['a']
    b_pre = par.comb['c_pre']['b']

    de_nom = 1 - q_fuel_pot * (gamma * a_pre + (1 - gamma) * a_dif)

    dq_comb_dphi = ((q_fuel_pot * dt_dphi * (gamma * b_pre +
                     (1 - gamma) * b_dif)) / de_nom)

    # Smooth max(dq, 0)
    eps = 1e-4
    dq_comb_dphi = 0.5 * dq_comb_dphi + 0.5 * ca.sqrt(eps**2 + dq_comb_dphi**2)

    return dq_comb_dphi
