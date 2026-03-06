"""In-cylinder model helper functions"""

import numpy as np
from typing import Tuple, Dict, Any, Union

try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False


def cyl_vol(ca: float, eng: Dict[str, float]) -> Tuple[float, float, float]:
    """Calculate cylinder volume and its derivative
    
    Args:
        ca: Crank angle [degCA] with TDC at ca = 0
        eng: Engine geometry parameters dictionary
    
    Returns:
        v: Cylinder volume [m^3]
        dv_deg: Cylinder volume derivative [m^3/deg]
        stroke: Piston stroke length [m]
    """
    # Define ca in radians
    ca_rad = ca * np.pi / 180
    
    # Cylinder Volume
    v = (eng['vol_dis'] / (eng['epsilon'] - 1) +
         eng['vol_dis'] * 0.5 *
         ((eng['length_con_rod'] + eng['rad_crank']) / eng['rad_crank'] -
          np.sqrt(eng['length_con_rod']**2 / eng['rad_crank']**2 - np.sin(ca_rad)**2) -
          np.cos(ca_rad)))
    
    # Cylinder Volume Derivative
    dv_rad = (eng['vol_dis'] * 0.5 *
             (np.sin(ca_rad) +
              (eng['length_con_rod']**2 / eng['rad_crank']**2 - np.sin(ca_rad)**2)**(-0.5) *
              np.sin(ca_rad) * np.cos(ca_rad)))
    dv_deg = dv_rad * np.pi / 180
    
    # Piston Stroke Length
    rod_ratio = eng['rad_crank'] / eng['length_con_rod']
    f = (1 + 1 / rod_ratio - np.cos(ca_rad) -
         1 / rod_ratio * np.sqrt(1 - rod_ratio**2 * np.sin(ca_rad)**2))
    stroke = eng['rad_crank'] * f
    
    return v, dv_deg, stroke


def calc_kappa(xi: Dict[str, float], theta: float, 
              spec_r: Dict[str, float]) -> float:
    """Calculate heat capacity ratio (kappa) for gas mixture
    
    Uses polynomial coefficients from GRI-Mech 3.0 database to calculate
    temperature-dependent specific heat capacities.
    
    Args:
        xi: Species mass fractions dictionary
        theta: Temperature [K]
        spec_r: Specific gas constants dictionary [J/(kg*K)]
    
    Returns:
        kappa_tot: Heat capacity ratio [-]
    """
    # Polynomial coefficients for each species
    # Source: http://combustion.berkeley.edu/gri-mech/version30/files30/thermo30.dat
    # 1:7  --> a1 - a7 for theta  > 1000 (high temperature regime)
    # 8:14 --> a1 - a7 for theta <= 1000 (low temperature regime)
    
    coef = {
        'N2': np.array([
            [0.02926640E+02, 0.14879768E-02, -0.05684760E-05, 0.10097038E-09,
             -0.06753351E-13, -0.09227977E+04, 0.05980528E+02],
            [0.03298677E+02, 0.14082404E-02, -0.03963222E-04, 0.05641515E-07,
             -0.02444854E-10, -0.10208999E+04, 0.03950372E+02]
        ]),
        'O2': np.array([
            [3.28253784E+00, 1.48308754E-03, -7.57966669E-07, 2.09470555E-10,
             -2.16717794E-14, -1.08845772E+03, 5.45323129E+00],
            [3.78245636E+00, -2.99673416E-03, 9.84730201E-06, -9.68129509E-09,
             3.24372837E-12, -1.06394356E+03, 3.65767573E+00]
        ]),
        'CO2': np.array([
            [3.85746029E+00, 4.41437026E-03, -2.21481404E-06, 5.23490188E-10,
             -4.72084164E-14, -4.87591660E+04, 2.27163806E+00],
            [2.35677352E+00, 8.98459677E-03, -7.12356269E-06, 2.45919022E-09,
             -1.43699548E-13, -4.83719697E+04, 9.90105222E+00]
        ]),
        'H2O': np.array([
            [3.03399249E+00, 2.17691804E-03, -1.64072518E-07, -9.70419870E-11,
             1.68200992E-14, -3.00042971E+04, 4.96677010E+00],
            [4.19864056E+00, -2.03643410E-03, 6.52040211E-06, -5.48797062E-09,
             1.77197817E-12, -3.02937267E+04, -8.49032208E-01]
        ]),
        'CxHy': np.array([
            [0.22818893E+02, 0.32543454E-01, -0.11120041E-04, 0.17131743E-08,
             -0.96212101E-13, -0.33678738E+05, -0.94335007E+02],
            [0.30149546E+01, 0.54457203E-01, 0.21812681E-04, -0.54234111E-07,
             0.20808730E-10, -0.26003379E+05, 0.17508575E+02]
        ])
    }
    
    cp_tot = 0
    spec_r_tot = 0
    
    for species in xi.keys():
        if species not in coef:
            continue
            
        # Select coefficients based on temperature
        if theta > 1000:
            a_k = coef[species][0, :5]
        else:
            a_k = coef[species][1, :5]
        
        # Calculate cp contribution
        cp_species = (a_k[0] + a_k[1] * theta + a_k[2] * theta**2 +
                     a_k[3] * theta**3 + a_k[4] * theta**4)
        
        cp_tot += cp_species * spec_r[species] * xi[species]
        spec_r_tot += spec_r[species] * xi[species]
    
    cv_tot = cp_tot - spec_r_tot
    kappa_tot = cp_tot / cv_tot
    
    return kappa_tot


def calc_kappa_casadi(xi: Dict[str, Union[float, 'ca.SX', 'ca.MX']], 
                     theta: Union[float, 'ca.SX', 'ca.MX'],
                     spec_r: Dict[str, float]) -> Union[float, 'ca.SX', 'ca.MX']:
    """Calculate heat capacity ratio (kappa) for gas mixture using CasADi
    
    Uses polynomial coefficients from GRI-Mech 3.0 database to calculate
    temperature-dependent specific heat capacities. This version uses CasADi's
    if_else for symbolic differentiation.
    
    Args:
        xi: Species mass fractions dictionary (can be CasADi symbolic)
        theta: Temperature [K] (can be CasADi symbolic)
        spec_r: Specific gas constants dictionary [J/(kg*K)]
    
    Returns:
        kappa_tot: Heat capacity ratio [-] (CasADi symbolic if inputs are symbolic)
    """
    if not CASADI_AVAILABLE:
        raise ImportError("CasADi is not available. Install it to use calc_kappa_casadi.")
    
    # Polynomial coefficients for each species
    # Source: http://combustion.berkeley.edu/gri-mech/version30/files30/thermo30.dat
    # 1:7  --> a1 - a7 for theta  > 1000 (high temperature regime)
    # 8:14 --> a1 - a7 for theta <= 1000 (low temperature regime)
    
    coef = {
        'N2': np.array([
            [0.02926640E+02, 0.14879768E-02, -0.05684760E-05, 0.10097038E-09,
             -0.06753351E-13, -0.09227977E+04, 0.05980528E+02],
            [0.03298677E+02, 0.14082404E-02, -0.03963222E-04, 0.05641515E-07,
             -0.02444854E-10, -0.10208999E+04, 0.03950372E+02]
        ]),
        'O2': np.array([
            [3.28253784E+00, 1.48308754E-03, -7.57966669E-07, 2.09470555E-10,
             -2.16717794E-14, -1.08845772E+03, 5.45323129E+00],
            [3.78245636E+00, -2.99673416E-03, 9.84730201E-06, -9.68129509E-09,
             3.24372837E-12, -1.06394356E+03, 3.65767573E+00]
        ]),
        'CO2': np.array([
            [3.85746029E+00, 4.41437026E-03, -2.21481404E-06, 5.23490188E-10,
             -4.72084164E-14, -4.87591660E+04, 2.27163806E+00],
            [2.35677352E+00, 8.98459677E-03, -7.12356269E-06, 2.45919022E-09,
             -1.43699548E-13, -4.83719697E+04, 9.90105222E+00]
        ]),
        'H2O': np.array([
            [3.03399249E+00, 2.17691804E-03, -1.64072518E-07, -9.70419870E-11,
             1.68200992E-14, -3.00042971E+04, 4.96677010E+00],
            [4.19864056E+00, -2.03643410E-03, 6.52040211E-06, -5.48797062E-09,
             1.77197817E-12, -3.02937267E+04, -8.49032208E-01]
        ]),
        'CxHy': np.array([
            [0.22818893E+02, 0.32543454E-01, -0.11120041E-04, 0.17131743E-08,
             -0.96212101E-13, -0.33678738E+05, -0.94335007E+02],
            [0.30149546E+01, 0.54457203E-01, 0.21812681E-04, -0.54234111E-07,
             0.20808730E-10, -0.26003379E+05, 0.17508575E+02]
        ])
    }
    
    cp_tot = 0
    spec_r_tot = 0
    
    for species in xi.keys():
        if species not in coef:
            continue
        
        # Use CasADi's if_else for symbolic conditional
        # Select coefficients based on temperature
        coef_high = coef[species][0, :5]
        coef_low = coef[species][1, :5]
        
        # Build a_k using if_else for each coefficient
        a_k = ca.vertcat(*[
            ca.if_else(theta > 1000, coef_high[i], coef_low[i])
            for i in range(5)
        ])
        
        # Calculate cp contribution
        cp_species = (a_k[0] + a_k[1] * theta + a_k[2] * theta**2 +
                     a_k[3] * theta**3 + a_k[4] * theta**4)
        
        cp_tot += cp_species * spec_r[species] * xi[species]
        spec_r_tot += spec_r[species] * xi[species]
    
    cv_tot = cp_tot - spec_r_tot
    kappa_tot = cp_tot / cv_tot
    
    return kappa_tot


def static_cylinder_conditions(q_comb: float, p_cyl: float, v_cyl: float,
                               par_model: Any, par_op: Any,
                               num_sym: str = 'Num') -> Tuple[float, float, float, float, float, float, float]:
    """Calculate static cylinder conditions (temperature, kappa, species)
    
    Args:
        q_comb: Cumulative heat release [J]
        p_cyl: Cylinder pressure [Pa]
        v_cyl: Cylinder volume [m^3]
        par_model: Model parameters
        par_op: Operating point parameters
        num_sym: 'Num' for numerical, 'Sym' for symbolic computation
    
    Returns:
        kappa: Heat capacity ratio [-]
        spec_r: Specific gas constant [J/(kg*K)]
        theta_cyl: Cylinder temperature [K]
        xi_o2: Oxygen mass fraction [-]
        x_bg: Burnt gas mass fraction [-]
        x_bz: Burned zone mass fraction [-]
        zeta_comb: Conversion rate [-]
    """
    species_list = ['N2', 'O2', 'CO2', 'H2O', 'CxHy']
    
    # Calculate Burned Fuel Mass and Mols
    m_fuel_conv = q_comb / par_model.thermo['fuel']['low_heat_val']
    mol_fuel_conv = m_fuel_conv / par_model.thermo['gas']['molar_mass']['CxHy']
    
    # Calculate species masses [kg]
    m_cyl = {}
    m_cyl_tot = 0
    
    for s in species_list[:-1]:  # Exclude CxHy
        m_cyl[s] = (par_model.thermo['fuel']['nu'][s] * mol_fuel_conv *
                   par_model.thermo['gas']['molar_mass'][s] +
                   par_op.m_cyl_tot * getattr(par_op, f'xi_{s}', 0))
        m_cyl_tot += m_cyl[s]
    
    m_cyl['CxHy'] = 0
    
    # Calculate air and burnt gas share
    m_cyl['air'] = (par_model.thermo['fuel']['nu']['air'] * mol_fuel_conv *
                   par_model.thermo['gas']['molar_mass']['air'] +
                   par_op.m_cyl_tot * par_op.xi_air)
    m_cyl['bg'] = m_cyl_tot - m_cyl['air']
    
    # Calculate Species Mass Fractions [-]
    xi = {}
    for s in species_list:
        xi[s] = m_cyl[s] / m_cyl_tot
    
    xi_air = m_cyl['air'] / m_cyl_tot
    xi_bg = 1 - xi_air
    
    # Calculate Specific Gas Constant [J/(kg*K)]
    spec_r = 0
    for s in species_list:
        spec_r += xi[s] * par_model.thermo['gas']['spec_r'][s]
    
    # Temperature [K]
    theta_cyl = p_cyl * v_cyl / (m_cyl_tot * spec_r)
    
    # Kappa update
    if num_sym == 'Sym':
        kappa = calc_kappa_casadi(xi, theta_cyl, par_model.thermo['gas']['spec_r'])
    else:
        kappa = calc_kappa(xi, theta_cyl, par_model.thermo['gas']['spec_r'])
    
    # Oxygen mass fraction
    xi_o2 = xi['O2']
    x_bg = xi_bg
    
    # Burned zone mass fraction
    x_bz = (q_comb * (par_model.thermo['fuel']['air_fuel_ratio_st'] / 
            (1 - par_op.xi_bg) + 1) /
            (m_cyl_tot * par_model.thermo['fuel']['low_heat_val'] + q_comb))
    
    # Conversion rate
    zeta_comb = q_comb / (m_cyl_tot * par_model.thermo['fuel']['low_heat_val'])
    
    return kappa, spec_r, theta_cyl, xi_o2, x_bg, x_bz, zeta_comb


def dynamic_cylinder_conditions(ca: float, p_cyl: float, dq_comb: float,
                                v_cyl: float, dv_cyl: float, stroke: float,
                                theta_cyl: float, kappa: float, spec_r: float,
                                par_model: Any, par_op: Any) -> Tuple[float, float, float]:
    """Calculate dynamic cylinder condition derivatives
    
    Args:
        ca: Crank angle [degCA]
        p_cyl: Cylinder pressure [Pa]
        dq_comb: Heat release rate [J/degCA]
        v_cyl: Cylinder volume [m^3]
        dv_cyl: Cylinder volume derivative [m^3/degCA]
        stroke: Piston stroke [m]
        theta_cyl: Cylinder temperature [K]
        kappa: Heat capacity ratio [-]
        spec_r: Specific gas constant [J/(kg*K)]
        par_model: Model parameters
        par_op: Operating point parameters
    
    Returns:
        dp_cyl: Cylinder pressure derivative [Pa/degCA]
        dimep: IMEP derivative [Pa/degCA]
        dq_wall: Wall heat loss rate [J/degCA]
    """
    # Derive Wall Heat Transfer [J/degCA]
    if par_model.opts['enable_wall_heat_loss']:
        dq_wall = derive_wall_heat_transfer(
            ca, stroke, theta_cyl, p_cyl,
            par_op.theta_ivc, par_op.p_int,
            par_op.v_int, v_cyl, par_op.eng_spd, par_model)
    else:
        dq_wall = 0
    
    # Derive Cylinder Pressure Derivative [Pa/degCA]
    dm_fuel = 0  # Assuming instantaneous combustion
    dp_cyl = (-kappa * p_cyl * dv_cyl / v_cyl +
              spec_r * theta_cyl * dm_fuel / v_cyl +
              (kappa - 1) / v_cyl * (dq_wall + dq_comb))
    
    # Derive Indicated Mean Effective Pressure Derivative [Pa/degCA]
    dimep = p_cyl * dv_cyl / par_model.eng['vol_dis']
    
    return dp_cyl, dimep, dq_wall


def derive_wall_heat_transfer(ca: float, stroke: float,
                              theta_cyl: float, p_cyl: float,
                              theta_ivc: float, p_ivc: float,
                              vol_ivc: float, vol: float,
                              eng_spd: float, par_model: Any) -> float:
    """Calculate wall heat transfer rate
    
    Uses Woschni heat transfer model.
    
    Args:
        ca: Crank angle [degCA]
        stroke: Piston stroke [m]
        theta_cyl: Cylinder temperature [K]
        p_cyl: Cylinder pressure [Pa]
        theta_ivc: Temperature at IVC [K]
        p_ivc: Pressure at IVC [Pa]
        vol_ivc: Volume at IVC [m^3]
        vol: Current cylinder volume [m^3]
        eng_spd: Engine speed [1/s]
        par_model: Model parameters
    
    Returns:
        dq_wall_dphi: Wall heat transfer rate [J/degCA]
    """
    # Transformation factor dPhidT in [degCA/s]
    dphi_dt = eng_spd * 360
    
    # Transformation factor dTdPhi in [s/degCA]
    dt_dphi = 1 / dphi_dt
    
    # Wall temperature in [K]
    air_excess_factor = 1
    theta_wall = (360 + 9 * air_excess_factor**0.4 *
                 np.sqrt(par_model.eng['bore'] * eng_spd * 60))
    
    # Surface area of the combustion chamber
    area_comb_chamber = (par_model.eng['area_cyl_head'] +
                        par_model.eng['area_bore'] +
                        np.pi * par_model.eng['bore'] * stroke)
    
    # Mean piston velocity cm in [m / s]
    cm = 4 * par_model.eng['rad_crank'] * eng_spd
    
    # Constant c1
    c1 = par_model.wall['c1']
    
    # Characteristic speed c in [m / s]
    # Estimated motored pressure
    p_cyl_mot = p_ivc * (vol_ivc / vol)**1.33
    delta_p_cyl = p_cyl - p_cyl_mot
    
    c2 = 5.0e-3 + 2.3e-5 * (theta_wall - 600)
    
    c = (c1 * cm + c2 * (par_model.eng['vol_dis'] * theta_ivc) /
         (p_ivc * vol_ivc) * delta_p_cyl * par_model.wall['scale_fired_whl'])
    
    # Heat transfer coefficient in [J / s*m^2*K]
    alpha = (0.013 * par_model.eng['bore']**(-0.2) * p_cyl**(0.8) *
            theta_cyl**(-0.53) * c**(0.8))
    
    # Scaling identified with compression curve
    alpha = alpha * par_model.wall['scale_whl']
    
    # Heat flow through cylinder wall in [J/s]
    dq_wall_dt = alpha * area_comb_chamber * (theta_wall - theta_cyl)
    
    # Heat flow through cylinder wall in [J/degCA]
    dq_wall_dphi = 0.8 * dq_wall_dt * dt_dphi
    
    return dq_wall_dphi
