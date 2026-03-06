"""Ignition delay Joerg model (CasADi version)"""

import casadi as ca


def ignition_delay_joerg(p, T, x_o2):
    """Calculate ignition delay using Joerg model (CasADi symbolic).

    Args:
        p: Cylinder pressure [Pa]
        T: Cylinder temperature [K]
        x_o2: Mass fraction of oxygen [0-1]

    Returns:
        tau_chem: Chemical ignition delay [ms]
    """
    p_ref = 40e5
    x_o2_ref = 0.224
    R = 1

    def func_arr(p, T, x_o2, c_pres, c_o2, k, ea):
        return k * (p / p_ref)**c_pres * (x_o2 / x_o2_ref)**c_o2 * ca.exp(ea / T / R)

    tau_ht = func_arr(p, T, x_o2, -1.05, -1.5, 1.25e-07, 15813)
    tau_lt = func_arr(p, T, x_o2, 0, 0, 1.09e-08, 13188)
    a_ht = func_arr(p, T, x_o2, -1, -1, 3.11e-03, 0)
    a_lt = func_arr(p, T, x_o2, 0, 0, 4.51e-12, 21079)

    w_ht = a_ht / (a_ht + a_lt)
    w_lt = 1 - w_ht

    tau_chem = w_ht * tau_ht + w_lt * tau_lt
    return tau_chem
