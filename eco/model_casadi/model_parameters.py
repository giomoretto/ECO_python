"""Model parameters for engine combustion simulation

This module defines all physical and model parameters for the engine
combustion simulation.
"""

import numpy as np
from typing import Optional, Dict, Any


class ModelParameters:
    """Engine model parameters
    
    This class contains all parameters needed for engine combustion modeling,
    including engine geometry, thermodynamic properties, wall heat loss model,
    ignition delay model, and combustion model parameters.
    """
    
    def __init__(self, x_sol: Optional[np.ndarray] = None):
        """Initialize model parameters
        
        Args:
            x_sol: Optional parameter vector for calibration
        """
        # Initialize structure
        self.eng = {}
        self.thermo = {}
        self.wall = {}
        self.ign = {}
        self.inj = {}
        self.comb = {}
        self.opts = {}
        
        self._initialize_engine_geometry()
        self._initialize_thermodynamics()
        self._initialize_wall_heat_loss()
        self._initialize_ignition_delay()
        self._initialize_combustion()
        self._initialize_injector()
        self._initialize_options()
        
        # Set default values for states and outputs
        self.n_states = 3
        self.n_outputs = 8
        self.n_inputs = 2
        
        if x_sol is not None:
            self._load_parameters(x_sol)
    
    def _initialize_engine_geometry(self):
        """Initialize engine geometry parameters"""
        # Total displacement volume in [m^3]
        self.eng['vol_dis_total'] = 1.968e-3
        # Number of cylinders in [1]
        self.eng['num_cyl'] = 4
        # Displacement volume per cylinder in [m^3]
        self.eng['vol_dis'] = self.eng['vol_dis_total'] / self.eng['num_cyl']
        # Compression ratio (epsilon) in [1]
        self.eng['epsilon'] = 16.5
        # Clearance volume per cylinder in [m^3]
        self.eng['vol_clear'] = self.eng['vol_dis'] / (self.eng['epsilon'] - 1)
        # Stroke in [m]
        self.eng['stroke'] = 95.5e-3
        # Crank radius in [m]
        self.eng['rad_crank'] = self.eng['stroke'] / 2
        # Con-rod length in [m]
        self.eng['length_con_rod'] = 3 * self.eng['rad_crank']
        # Bore area in [m^2]
        self.eng['area_bore'] = self.eng['vol_dis'] / self.eng['stroke']
        # Cylinder head area in [m^2]
        self.eng['area_cyl_head'] = self.eng['area_bore'] * 1.2
        # Bore diameter in [m]
        self.eng['bore'] = 2 * np.sqrt(self.eng['area_bore'] / np.pi)
    
    def _initialize_thermodynamics(self):
        """Initialize thermodynamic parameters"""
        self.thermo['psi'] = {'air': {}, 'bg': {}}
        self.thermo['fuel'] = {'nu': {}}
        self.thermo['gas'] = {'molar_mass': {}, 'spec_r': {}}
        self.thermo['xi'] = {'air': {}, 'bg': {}}
        
        # Species amount fractions for air in [1]
        self.thermo['psi']['air']['O2'] = 0.21
        self.thermo['psi']['air']['N2'] = 0.79
        
        # Fuel related parameters (Diesel)
        # Lower heating value in [J/kg]
        self.thermo['fuel']['low_heat_val'] = 42.6e6
        # Upper heating value in [J/kg]
        self.thermo['fuel']['up_heat_val'] = 45e6
        # C and H composition, C_x H_y
        self.thermo['fuel']['x'] = 10.8
        self.thermo['fuel']['y'] = 18.7
        
        # Stochiometric coefficients
        self.thermo['fuel']['nu']['N2'] = 0
        self.thermo['fuel']['nu']['O2'] = (-1) * (self.thermo['fuel']['x'] + 
                                                   self.thermo['fuel']['y'] / 4)
        self.thermo['fuel']['nu']['CO2'] = self.thermo['fuel']['x']
        self.thermo['fuel']['nu']['H2O'] = self.thermo['fuel']['y'] / 2
        self.thermo['fuel']['nu']['CxHy'] = -1
        self.thermo['fuel']['nu']['air'] = ((-1) / self.thermo['psi']['air']['O2'] *
                                            (self.thermo['fuel']['x'] + 
                                             self.thermo['fuel']['y'] / 4))
        self.thermo['fuel']['nu']['bg'] = 1
        
        # Density at 22°C in [kg/m^3]
        self.thermo['fuel']['rho'] = 834 * (1 + 0.00067 * 7) * 1
        # Kinematic fuel viscosity in [m^2/s]
        self.thermo['fuel']['kin_visc'] = 2.8e-6
        # Surface tension in [N/m]
        self.thermo['fuel']['surf_ten'] = 0.03
        # Evaporation temperature in [K]
        self.thermo['fuel']['theta_evap'] = 570
        # Evaporation enthalpy in [J/kg]
        self.thermo['fuel']['evap_enthalpy'] = 359 * 1e3
        # Spec. heat capacity in [J/(kg K)]
        self.thermo['fuel']['spec_heat_cap'] = 1926
        
        # In-Cylinder Gas related parameters
        # Species amount fractions for burnt gas in [1]
        mol_bg_tot = (self.thermo['fuel']['x'] + self.thermo['fuel']['y'] / 2 +
                     self.thermo['psi']['air']['N2'] / self.thermo['psi']['air']['O2'] *
                     (self.thermo['fuel']['x'] + self.thermo['fuel']['y'] / 4))
        
        self.thermo['psi']['bg']['H2O'] = (self.thermo['fuel']['y'] / 2) / mol_bg_tot
        self.thermo['psi']['bg']['CO2'] = self.thermo['fuel']['x'] / mol_bg_tot
        self.thermo['psi']['bg']['N2'] = (self.thermo['psi']['air']['N2'] / 
                                         self.thermo['psi']['air']['O2'] *
                                         (self.thermo['fuel']['x'] + 
                                          self.thermo['fuel']['y'] / 4)) / mol_bg_tot
        
        # Species Molar masses in [kg/mol]
        self.thermo['gas']['molar_mass']['C'] = 12.0108e-3
        self.thermo['gas']['molar_mass']['H'] = 1.0079e-3
        self.thermo['gas']['molar_mass']['O'] = 15.9994e-3
        self.thermo['gas']['molar_mass']['N'] = 14.0067e-3
        self.thermo['gas']['molar_mass']['O2'] = 2 * self.thermo['gas']['molar_mass']['O']
        self.thermo['gas']['molar_mass']['N2'] = 2 * self.thermo['gas']['molar_mass']['N']
        self.thermo['gas']['molar_mass']['H2O'] = (2 * self.thermo['gas']['molar_mass']['H'] +
                                                   self.thermo['gas']['molar_mass']['O'])
        self.thermo['gas']['molar_mass']['CO2'] = (self.thermo['gas']['molar_mass']['C'] +
                                                   2 * self.thermo['gas']['molar_mass']['O'])
        self.thermo['gas']['molar_mass']['air'] = (self.thermo['psi']['air']['O2'] * 
                                                   self.thermo['gas']['molar_mass']['O2'] +
                                                   self.thermo['psi']['air']['N2'] * 
                                                   self.thermo['gas']['molar_mass']['N2'])
        self.thermo['gas']['molar_mass']['fuel'] = (self.thermo['fuel']['x'] * 
                                                    self.thermo['gas']['molar_mass']['C'] +
                                                    self.thermo['fuel']['y'] * 
                                                    self.thermo['gas']['molar_mass']['H'])
        self.thermo['gas']['molar_mass']['CxHy'] = (self.thermo['fuel']['x'] * 
                                                    self.thermo['gas']['molar_mass']['C'] +
                                                    self.thermo['fuel']['y'] * 
                                                    self.thermo['gas']['molar_mass']['H'])
        self.thermo['gas']['molar_mass']['bg'] = (self.thermo['psi']['bg']['H2O'] * 
                                                  self.thermo['gas']['molar_mass']['H2O'] +
                                                  self.thermo['psi']['bg']['CO2'] * 
                                                  self.thermo['gas']['molar_mass']['CO2'] +
                                                  self.thermo['psi']['bg']['N2'] * 
                                                  self.thermo['gas']['molar_mass']['N2'])
        
        # Universal/molar gas constant in [J/(K*mol)]
        self.thermo['gas']['univ_r'] = 8.314
        
        # Specific gas constants in [J/K*kg]
        for species in ['O2', 'N2', 'H2O', 'CO2', 'air', 'fuel', 'CxHy', 'bg']:
            self.thermo['gas']['spec_r'][species] = (self.thermo['gas']['univ_r'] /
                                                     self.thermo['gas']['molar_mass'][species])
        
        # Species mass fractions for air in [1]
        self.thermo['xi']['air']['O2'] = (self.thermo['psi']['air']['O2'] * 
                                         self.thermo['gas']['molar_mass']['O2'] /
                                         self.thermo['gas']['molar_mass']['air'])
        self.thermo['xi']['air']['N2'] = (self.thermo['psi']['air']['N2'] * 
                                         self.thermo['gas']['molar_mass']['N2'] /
                                         self.thermo['gas']['molar_mass']['air'])
        
        # Species mass fractions for burnt gas [1]
        self.thermo['xi']['bg']['CO2'] = (self.thermo['psi']['bg']['CO2'] * 
                                         self.thermo['gas']['molar_mass']['CO2'] /
                                         self.thermo['gas']['molar_mass']['bg'])
        self.thermo['xi']['bg']['H2O'] = (self.thermo['psi']['bg']['H2O'] * 
                                         self.thermo['gas']['molar_mass']['H2O'] /
                                         self.thermo['gas']['molar_mass']['bg'])
        self.thermo['xi']['bg']['N2'] = (self.thermo['psi']['bg']['N2'] * 
                                        self.thermo['gas']['molar_mass']['N2'] /
                                        self.thermo['gas']['molar_mass']['bg'])
        
        # Stoichiometric air fuel ratio [kg air/kg fuel]
        self.thermo['fuel']['air_fuel_ratio_st'] = (
            (1 / self.thermo['psi']['air']['O2'] *
             (self.thermo['fuel']['x'] + self.thermo['fuel']['y'] / 4) *
             self.thermo['gas']['molar_mass']['air']) /
            self.thermo['gas']['molar_mass']['CxHy'])
    
    def _initialize_wall_heat_loss(self):
        """Initialize wall heat loss model parameters"""
        # Enable wall heat loss model
        self.opts['enable_wall_heat_loss'] = True
        # Enable variable wall heat loss
        self.opts['enable_variable_whl'] = False
        # Intake swirl number in [1]
        self.wall['int_swirl_number'] = 2.5
        # C1 factor in [1]
        self.wall['c1'] = 2.28 + 0.308 * self.wall['int_swirl_number']
        # Scale factor
        self.wall['scale_whl'] = 2.8
        self.wall['scale_fired_whl'] = 0.05
        # ca-variable wall heat losses
        self.wall['smoothness'] = 10
        self.wall['drop_percentage'] = 25
        self.wall['ca_drop_location'] = 0
    
    def _initialize_ignition_delay(self):
        """Initialize ignition delay model parameters"""
        # Select ignition delay model
        self.opts['ignition_delay_model'] = 'Joerg'  # 'Joerg', 'Nils'
        
        # Physical ignition delay modeled as constant in [s]
        self.ign['phys'] = {'const': 2.5e-4}
        
        # Minimal and maximal ignition delay
        self.ign['tau_min'] = 0
        self.ign['tau_max'] = 2e-3
        
        # Correction factor for physical ignition delay
        self.ign['chem'] = {'const': 1}
    
    def _initialize_combustion(self):
        """Initialize combustion model parameters"""
        # Parameters for Simplified Model
        self.comb['c_dif'] = {'a': 5.27e-4, 'b': 2.07e3}
        self.comb['c_pre'] = {'a': 0, 'b': 3.8e3}
        self.comb['gamma'] = {'tau_min': 2.7e-4, 'tau_max': 2.9e-4}
        # Combustion efficiency in [1]
        self.comb['eta_comb'] = 1
    
    def _initialize_injector(self):
        """Initialize injector model parameters"""
        # Placeholder for injector model parameters
        pass
    
    def _initialize_options(self):
        """Initialize simulation options"""
        # NOx model will be set later if needed
        self.nox_model = None
    
    def _load_parameters(self, x_sol: np.ndarray):
        """Load parameters from calibration vector
        
        Args:
            x_sol: Calibration parameter vector
        """
        self.comb['c_dif']['a'] = x_sol[0]
        self.comb['c_dif']['b'] = x_sol[1]
        self.comb['c_pre']['a'] = x_sol[2]
        self.comb['c_pre']['b'] = x_sol[3]
        self.ign['phys']['const'] = x_sol[4]
        self.comb['gamma']['tau_min'] = x_sol[5]
        self.comb['gamma']['tau_max'] = x_sol[6]
        self.ign['tau_min'] = x_sol[7]
        self.ign['tau_max'] = x_sol[8]
