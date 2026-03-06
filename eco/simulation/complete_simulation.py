"""Complete simulation with combustion model"""

import numpy as np
from typing import Dict, Any
import sys
sys.path.insert(0, '/home/morettog/projects/phd/python_code')
from eco.model.complete_model import complete_model


def complete_simulation(ca: np.ndarray, x0: np.ndarray, u: np.ndarray,
                       par_sim: Any, par_model: Any) -> Dict[str, np.ndarray]:
    """Run complete engine simulation with combustion
    
    Args:
        ca: Crank angle vector [degCA]
        x0: Initial state vector
        u: Input trajectory [SOE [degCA], DOE [µs]] (nInputs x nTimeSteps)
        par_sim: Simulation parameters
        par_model: Model parameters
    
    Returns:
        Dictionary with simulation results (ca, x, u, y, xdot)
    """
    # Initialize vectors
    x = np.zeros((par_model.n_states, len(ca)))
    y = np.zeros((par_model.n_outputs, len(ca)))
    xdot = np.zeros((par_model.n_states, len(ca)))
    x[:, 0] = x0
    
    # Run simulation
    for kk in range(len(ca)):
        x_new = x[:, kk]
        ca_act = ca[kk]
        
        # Get current delta_phi
        if kk != len(ca) - 1:
            delta_phi = (ca[kk + 1] - ca[kk]) / par_sim.opts['n_int']
        else:
            delta_phi = (ca[kk] - ca[kk - 1]) / par_sim.opts['n_int']
        
        # Do integration
        for jj in range(par_sim.opts['n_int']):
            if par_sim.opts['int'] == 'EulerFW':
                if jj == 0:
                    x_new, xdot[:, kk], y[:, kk] = _euler_fw_integration_step(
                        x_new, u[:, kk], ca_act, par_model, par_sim.op, delta_phi)
                else:
                    x_new, _, _ = _euler_fw_integration_step(
                        x_new, u[:, kk], ca_act, par_model, par_sim.op, delta_phi)
            
            elif par_sim.opts['int'] == 'RK4' or par_sim.opts['int'] == 1:
                if jj == 0:
                    x_new, xdot[:, kk], y[:, kk] = _rk4_integration_step(
                        x_new, u[:, kk], ca_act, par_model, par_sim.op, delta_phi)
                else:
                    x_new, _, _ = _rk4_integration_step(
                        x_new, u[:, kk], ca_act, par_model, par_sim.op, delta_phi)
            
            ca_act = ca_act + delta_phi
        
        if kk != len(ca) - 1:
            x[:, kk + 1] = x_new
    
    # Write results to dict
    simout = {
        'ca': ca,
        'x': x,
        'u': u,
        'y': y,
        'xdot': xdot
    }
    
    return simout


def _euler_fw_integration_step(x_act: np.ndarray, u_act: np.ndarray,
                               ca_act: float, par_model: Any, par_op: Any,
                               delta_phi: float) -> tuple:
    """Euler forward integration step"""
    xdot, y = complete_model(x_act, u_act, ca_act, par_model, par_op, 'Num')
    x_new = x_act + xdot * delta_phi
    return x_new, xdot, y


def _rk4_integration_step(x_act: np.ndarray, u_act: np.ndarray,
                         ca_act: float, par_model: Any, par_op: Any,
                         delta_phi: float) -> tuple:
    """Runge-Kutta 4th order integration step"""
    # Define k1
    k1, y = complete_model(x_act, u_act, ca_act, par_model, par_op, 'Num')
    
    # Calculate k2
    x_k1 = x_act + k1 * delta_phi / 2
    k2, _ = complete_model(x_k1, u_act, ca_act + delta_phi / 2,
                          par_model, par_op, 'Num')
    
    # Calculate k3
    x_k2 = x_act + k2 * delta_phi / 2
    k3, _ = complete_model(x_k2, u_act, ca_act + delta_phi / 2,
                          par_model, par_op, 'Num')
    
    # Calculate k4
    x_k3 = x_act + k3 * delta_phi
    k4, _ = complete_model(x_k3, u_act, ca_act + delta_phi,
                          par_model, par_op, 'Num')
    
    # Calculate new x
    x_new = x_act + delta_phi / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return x_new, k1, y
