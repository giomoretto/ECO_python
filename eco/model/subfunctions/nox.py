"""NOx model helper functions"""

import numpy as np
from typing import Tuple, Any
from .ignition_delay import saturate_input


def two_zone_model(q_comb: float, theta_uz0: float, dp_cyl: float, p_cyl: float,
                   dv_cyl: float, v_cyl: float, t_cyl: float, kappa: float,
                   m_tot: float, par_op: Any, par_model: Any,
                   x_bg: float, x_bz: float) -> Tuple[float, float]:
    """Two-zone model for calculating burned gas temperature
    
    This model assumes two zones: unburned and burned. It calculates the
    unburned zone temperature under isentropic compression/expansion assumption.
    
    Args:
        q_comb: Cumulative heat release [J]
        theta_uz0: Unburned zone temperature [K]
        dp_cyl: Cylinder pressure derivative [Pa/degCA]
        p_cyl: Cylinder pressure [Pa]
        dv_cyl: Volume derivative [m^3/degCA]
        v_cyl: Cylinder volume [m^3]
        t_cyl: Mean cylinder temperature [K]
        kappa: Heat capacity ratio [-]
        m_tot: Total mass [kg]
        par_op: Operating point parameters
        par_model: Model parameters
        x_bg: Burnt gas mass fraction [-]
        x_bz: Burned zone mass fraction [-]
    
    Returns:
        d_theta_uz: Unburned zone temperature derivative [K/degCA]
        theta_bz: Burned zone temperature [K]
    """
    # Limit ThetaUZ to TCyl at current shooting node
    theta_uz0 = saturate_input(theta_uz0, 0, t_cyl)
    
    d_theta = t_cyl * (dp_cyl / p_cyl + dv_cyl / v_cyl)
    zeta_comb = q_comb / (m_tot * par_model.thermo['fuel']['low_heat_val'])
    
    q_min = 0.01
    q_max = 0.1
    u_comb = (saturate_input(zeta_comb, q_min, q_max) - q_min) / (q_max - q_min)
    
    # Unburned zone temperature
    d_theta_uz = 1 / p_cyl * dp_cyl * theta_uz0 * (kappa - 1) / kappa
    d_theta_uz = (1 - u_comb) * d_theta + u_comb * d_theta_uz
    
    # Burned zone temperature
    x_bz_sat = saturate_input(x_bz, 0.01, 1)
    theta_bz = u_comb * (t_cyl - theta_uz0) / x_bz_sat + theta_uz0
    
    return d_theta_uz, theta_bz


def nox_model(q_comb: float, theta_uz: float, theta_bz: float, no: float,
             m_tot: float, par_op: Any, par_model: Any, x_bg: float,
             v_cyl: float, dv_cyl: float, p_cyl: float, dp_cyl: float,
             theta_cyl: float) -> Tuple[float, float, float]:
    """Calculate NOx formation rate using extended Zeldovich mechanism
    
    Args:
        q_comb: Cumulative heat release [J]
        theta_uz: Unburned zone temperature [K]
        theta_bz: Burned zone temperature [K]
        no: NO mole fraction in burned zone [-]
        m_tot: Total injected mass [kg]
        par_op: Operating point parameters
        par_model: Model parameters
        x_bg: Burnt gas mass fraction [-]
        v_cyl: Cylinder volume [m^3]
        dv_cyl: Volume derivative [m^3/degCA]
        p_cyl: Cylinder pressure [Pa]
        dp_cyl: Pressure derivative [Pa/degCA]
        theta_cyl: Mean cylinder temperature [K]
    
    Returns:
        d_no_dphi: NO formation rate [1/degCA]
        no_ppm: NO concentration [ppm]
        no_eq: NO equilibrium concentration [mol/cm^3]
    """
    # Time passed per °CA
    dt_dphi = 1 / (par_op.eng_spd * 360)
    
    # Mole fraction to concentration in BG (mol/cm^3)
    mf2c = 1e-6 * p_cyl / (par_model.thermo['gas']['univ_r'] * theta_bz)
    
    # Saturate burned zone temperature
    theta_bz_sat = saturate_input(theta_bz, 1600, 3300)
    
    # Reaction rates
    k1f, k2f, k3f = _reaction_rates(theta_bz_sat)
    
    # Equilibrium concentrations
    x_n = _x_n_eq(theta_bz_sat)
    x_o = _x_o_eq(theta_bz_sat)
    x_n2 = _x_n2_eq(theta_bz_sat)
    x_o2 = _x_o2_eq(theta_bz_sat)
    x_oh = _x_oh_eq(theta_bz_sat)
    x_no = _x_no_eq(theta_bz_sat)
    no_eq = x_no * mf2c
    
    r1 = k1f * x_o * x_n2 * mf2c**2
    r2 = k2f * x_n * x_o2 * mf2c**2
    r3 = k3f * x_n * x_oh * mf2c**2
    
    q = r1 / (r2 + r3)
    
    # Turn on NOx model after combustion has started
    zeta_comb = q_comb / (m_tot * par_model.thermo['fuel']['low_heat_val'])
    
    q_min = 0.01
    q_max = 0.05
    u_nox = (saturate_input(zeta_comb, q_min, q_max) - q_min) / (q_max - q_min)
    
    # NOx formation rate
    no_scale = no / no_eq
    d_no_dt = u_nox * 2 * r1 * (1 - no_scale**2) / (1 + no_scale * q)
    
    # Include volume change
    lhv = par_model.thermo['fuel']['low_heat_val']
    sigma = par_model.thermo['fuel']['air_fuel_ratio_st']
    m_uz = par_op.m_cyl_tot - q_comb / lhv * sigma * (1 / (1 - par_op.xi_bg))
    v_uz = m_uz * par_op.spec_r_ivc * theta_uz / p_cyl
    v_bz = v_cyl - v_uz
    dv_uz = -dp_cyl * v_uz / par_op.kappa_ivc / p_cyl
    dv_bz = u_nox * (dv_cyl - dv_uz)
    d_no_dv = u_nox * no * dv_bz / v_bz
    d_no_dphi = d_no_dt * dt_dphi - d_no_dv
    
    # Convert NO concentration to ppm in whole gas
    n_tot = p_cyl * v_cyl / (par_model.thermo['gas']['univ_r'] * theta_cyl)
    no_ppm = 1e6 * no * (1e6 * v_bz) / n_tot
    
    return d_no_dphi, no_ppm, no_eq


def _reaction_rates(theta_bz: float) -> Tuple[float, float, float]:
    """Calculate reaction rates for extended Zeldovich mechanism
    
    Args:
        theta_bz: Burned zone temperature [K]
    
    Returns:
        k1f, k2f, k3f: Forward reaction rates
    """
    k1f = 1.47e13 * theta_bz**0.3 * np.exp(-37885.8746 / theta_bz)
    k2f = 6.40e09 * theta_bz**1.0 * np.exp(-3163.169283 / theta_bz)
    k3f = 3.80e13
    
    return k1f, k2f, k3f


def _x_n_eq(theta_bz: float) -> float:
    """Calculate equilibrium mole fraction of N"""
    coef = np.array([-93.5994804475774, 0.0635582925091314,
                    -1.78636544365512e-05, 1.86389452885879e-09])
    regr = np.array([1, theta_bz, theta_bz**2, theta_bz**3])
    return np.exp(np.dot(coef, regr))


def _x_n2_eq(theta_bz: float) -> float:
    """Calculate equilibrium mole fraction of N2"""
    coef = np.array([-0.216167752677653, -0.000141506777271925,
                    8.06135838153914e-08, -1.54929244598273e-11])
    regr = np.array([1, theta_bz, theta_bz**2, theta_bz**3])
    return np.exp(np.dot(coef, regr))


def _x_o_eq(theta_bz: float) -> float:
    """Calculate equilibrium mole fraction of O"""
    coef = np.array([-63.6851353343809, 0.0451581256769548,
                    -1.25158990443408e-05, 1.27610634104909e-09])
    regr = np.array([1, theta_bz, theta_bz**2, theta_bz**3])
    return np.exp(np.dot(coef, regr))


def _x_o2_eq(theta_bz: float) -> float:
    """Calculate equilibrium mole fraction of O2"""
    coef = np.array([-33.6431381400466, 0.0227657004453811,
                    -6.04189495663701e-06, 5.68577263026475e-10])
    regr = np.array([1, theta_bz, theta_bz**2, theta_bz**3])
    return np.exp(np.dot(coef, regr))


def _x_oh_eq(theta_bz: float) -> float:
    """Calculate equilibrium mole fraction of OH"""
    coef = np.array([-38.9129885343657, 0.0269739906905952,
                    -7.40667321236856e-06, 7.42212500799537e-10])
    regr = np.array([1, theta_bz, theta_bz**2, theta_bz**3])
    return np.exp(np.dot(coef, regr))


def _x_no_eq(theta_bz: float) -> float:
    """Calculate equilibrium mole fraction of NO"""
    coef = np.array([-34.406106632447, 0.0234949344561571,
                    -6.41287032472529e-06, 6.35111932336585e-10])
    regr = np.array([1, theta_bz, theta_bz**2, theta_bz**3])
    return np.exp(np.dot(coef, regr))


def no_equilibrium(t_burned: float, lambda_val: float = 1.0) -> float:
    """Calculate NO equilibrium concentration
    
    Args:
        t_burned: Burned zone temperature [K]
        lambda_val: Air-fuel equivalence ratio [-]
    
    Returns:
        NO equilibrium mole fraction in burned gas [-]
    """
    y0 = 0.14343
    c1 = 21.57
    c2 = -0.5528
    c3 = -1.846
    
    y = y0 * np.exp(-c1 * (t_burned / 1000 - c2)**c3)
    
    return y
