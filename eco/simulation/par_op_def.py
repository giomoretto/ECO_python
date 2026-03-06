"""Operating point definition

This module defines the operating point parameters for engine simulation.
"""

import numpy as np
from typing import Optional, Dict, Any
import sys
sys.path.insert(0, '/home/morettog/projects/phd/python_code')
from eco.model.subfunctions.in_cylinder import cyl_vol, static_cylinder_conditions


class OperatingPoint:
    """Operating point parameters for engine simulation"""
    
    def __init__(self, par_model: Any, meas: Optional[Dict] = None, 
                 num_sym: str = 'Num'):
        """Initialize operating point
        
        Args:
            par_model: Model parameters
            meas: Optional measurement dictionary
            num_sym: 'Num' for numerical, 'Sym' for symbolic
        """
        if meas is not None:
            # Use measurements
            self.eng_spd = meas['ne']  # 1/s
            self.p_im = meas['p_im']  # Pa
            self.theta_im = meas['t_im']  # K
            self.xi_bg_im = meas['x_bg']  # -
            self.p_rail = meas['p_rail']  # Pa
        else:
            # Default values
            self.eng_spd = 2000 / 60  # 1/s; 2000 1/min
            self.p_im = 1.2 * 1e5  # Pa 1.2bar
            self.theta_im = 318.15  # K 45°C
            self.xi_bg_im = 0.1  # -
            self.p_rail = 1000e5  # Pa 1000bar
        
        # Assumptions on exhaust and residual gas
        self.xi_res = 0.05  # -
        self.lambda_exh = 2
        
        # NOT USED
        self.p_exh = self.p_im + 0.3e5  # Pa
        self.theta_exh = 430 + 273.15  # K
        
        # Valve timings
        self.ca_ivc = -172  # degCAaTDC
        self.ca_evo = 155  # degCAaTDC
        self.ca_evc = 345  # degCAaTDC
        
        # Calculate cylinder conditions
        self._calculate_conditions(par_model, num_sym)
    
    def _calculate_conditions(self, par_model: Any, num_sym: str):
        """Calculate in-cylinder conditions at IVC"""
        # Volume at IVC
        self.v_int, _, _ = cyl_vol(self.ca_ivc, par_model.eng)
        
        # In-cylinder pressure at IVC
        self.p_int = self._gas_dynamic_effects(self.p_im)
        
        # Obtain fresh charge into cylinder and total mass
        m_beta = self._calc_m_beta(self.v_int)
        
        # Account for residual gas fraction in total mass
        self.m_cyl_tot = m_beta / (1 - self.xi_res)
        
        # Obtain residual mass in cylinder
        m_res_cyl = self.m_cyl_tot * self.xi_res
        
        # Exhaust Gas Burnt Gas & Air Ratio Determination
        xi_exh_bg = 1 / self.lambda_exh
        xi_exh_air = 1 - xi_exh_bg
        
        # Calculate composition of gas at Intake Valve Closing (IVC)
        # Fresh Air Mass
        m_beta_air = m_beta * (1 - self.xi_bg_im)
        # Fresh Burnt Gas
        m_beta_bg = m_beta - m_beta_air
        # Total fresh air
        m_cyl_air = m_beta_air + m_res_cyl * xi_exh_air
        m_cyl_bg = m_beta_bg + m_res_cyl * xi_exh_bg
        
        # In-cylinder species masses
        m_cyl_n2 = (par_model.thermo['xi']['air']['N2'] * m_cyl_air +
                   par_model.thermo['xi']['bg']['N2'] * m_cyl_bg)
        m_cyl_o2 = par_model.thermo['xi']['air']['O2'] * m_cyl_air
        m_cyl_cxhy = 0
        m_cyl_co2 = par_model.thermo['xi']['bg']['CO2'] * m_cyl_bg
        m_cyl_h2o = (self.m_cyl_tot - m_cyl_n2 - m_cyl_o2 - 
                    m_cyl_cxhy - m_cyl_co2)
        
        self.xi_air = m_cyl_air / self.m_cyl_tot
        self.xi_bg = 1 - self.xi_air
        
        self.xi_n2 = m_cyl_n2 / self.m_cyl_tot
        self.xi_o2 = m_cyl_o2 / self.m_cyl_tot
        self.xi_cxhy = m_cyl_cxhy / self.m_cyl_tot
        self.xi_co2 = m_cyl_co2 / self.m_cyl_tot
        self.xi_h2o = (1 - self.xi_n2 - self.xi_o2 - 
                      self.xi_cxhy - self.xi_co2)
        
        # Uppercase aliases (expected by static_cylinder_conditions)
        self.xi_N2 = self.xi_n2
        self.xi_O2 = self.xi_o2
        self.xi_CxHy = self.xi_cxhy
        self.xi_CO2 = self.xi_co2
        self.xi_H2O = self.xi_h2o
        
        # Calculate kappa and specific gas constant at IVC
        # Create a temporary operating point for static_cylinder_conditions
        class TempOP:
            pass
        temp_op = TempOP()
        temp_op.m_cyl_tot = self.m_cyl_tot
        temp_op.xi_air = self.xi_air
        temp_op.xi_bg = self.xi_bg
        temp_op.xi_N2 = self.xi_n2
        temp_op.xi_O2 = self.xi_o2
        temp_op.xi_CO2 = self.xi_co2
        temp_op.xi_H2O = self.xi_h2o
        temp_op.xi_CxHy = self.xi_cxhy
        
        self.kappa_ivc, self.spec_r_ivc, self.theta_ivc, _, _, _, _ = \
            static_cylinder_conditions(0, self.p_int, self.v_int, 
                                      par_model, temp_op, num_sym)
    
    def _calc_m_beta(self, v_int: float) -> float:
        """Calculate fresh charge mass
        
        Args:
            v_int: Volume at IVC [m^3]
        
        Returns:
            m_beta: Fresh charge mass [kg]
        """
        # Volumetric efficiency
        # Mean value for different pIVO/pIM
        lambda_lp = 0.96
        # Normalized for 45degC (318K) to one, measured for up to theta_im = 90 degC
        lambda_l_theta = (0.594 + 9.48e-4 * self.theta_im) / 0.8955
        # Measured only for rpm 2000
        lambda_l_omega = 0.8955
        # Combined volumetric efficiencies
        lambda_l = lambda_lp * lambda_l_theta * lambda_l_omega
        
        # Ideal gas constant
        R = 287
        # Density of gas in intake manifold
        rho = self.p_im / (R * self.theta_im)
        # Result
        m_beta = lambda_l * rho * v_int
        
        return m_beta
    
    def _gas_dynamic_effects(self, p_im: float) -> float:
        """Calculate pressure at IVC accounting for gas dynamic effects
        
        Args:
            p_im: Intake manifold pressure [Pa]
        
        Returns:
            p_ivc: Pressure at IVC [Pa]
        """
        # Everything in [Pa]
        p_ivc = -7122.9 + 1.1517 * p_im
        return p_ivc


def par_op_def(par_model: Any, meas: Optional[Dict] = None,
              num_sym: str = 'Num') -> OperatingPoint:
    """Define operating point parameters
    
    Args:
        par_model: Model parameters
        meas: Optional measurement dictionary
        num_sym: 'Num' for numerical, 'Sym' for symbolic
    
    Returns:
        Operating point object
    """
    return OperatingPoint(par_model, meas, num_sym)
