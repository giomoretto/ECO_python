#!/usr/bin/env python3
"""Simulate the complete engine model with acados and plot results."""

import sys
sys.path.insert(0, '/home/morettog/projects/phd/python_code')

import os
import ctypes
acados_path = '/home/morettog/projects/phd/acados'
os.environ.setdefault('ACADOS_SOURCE_DIR', acados_path)
acados_lib = os.path.join(acados_path, 'lib')
for lib in ['libblasfeo.so', 'libhpipm.so', 'libqpOASES_e.so', 'libacados.so']:
    ctypes.CDLL(os.path.join(acados_lib, lib), mode=ctypes.RTLD_GLOBAL)

import numpy as np
from eco.model_casadi.model_parameters import ModelParameters
from eco.simulation.par_op_def import OperatingPoint
from eco.simulation.acados_simulation import acados_simulation
from examples.utils import plot_results


def simulate(ca_start=-172, ca_end=155, delta_phi=0.5,
             soe=None, doe=None):
    """Simulate the complete model using the acados integrator.

    Args:
        ca_start: Start crank angle [degCA aTDC] (default: IVC = -172)
        ca_end: End crank angle [degCA aTDC] (default: EVO = 155)
        delta_phi: Crank angle step size [degCA]
        soe: Start of energizing per injection [degCA aTDC], array-like
        doe: Duration of energizing per injection [µs], array-like

    Returns:
        simout: Dictionary with simulation results (ca, x, u, y, xdot)
        par_op: Operating point parameters
        soe: SOE array
        doe: DOE array
    """
    if soe is None:
        soe = np.array([-15, 10])
    if doe is None:
        doe = np.array([300, 300])
    soe = np.atleast_1d(np.asarray(soe, dtype=float))
    doe = np.atleast_1d(np.asarray(doe, dtype=float))
    n_inj = len(soe)

    # --- Setup ---
    par_model = ModelParameters()
    par_model.n_states = 5
    par_model.n_outputs = 12
    par_model.n_inputs = 2 * n_inj

    par_op = OperatingPoint(par_model)

    class ParSim:
        pass
    par_sim = ParSim()
    par_sim.op = par_op
    par_sim.opts = {'delta_phi': delta_phi, 'int': 'RK4', 'n_int': 1}

    # Crank angle vector
    ca = np.arange(ca_start, ca_end + delta_phi, delta_phi)

    # Initial state: [p_cyl, q_comb, IMEP, theta_uz, NO]
    x0 = np.array([par_op.p_int, 0.0, 0.0, par_op.theta_ivc, 0.0])

    # Constant input over the whole range: [SOE1..SOEn, DOE1..DOEn]
    u_vec = np.concatenate([soe, doe])
    u = np.tile(u_vec, (len(ca), 1)).T

    # --- Simulate ---
    print(f"Running acados simulation: CA = [{ca_start}, {ca_end}] deg, "
          f"{n_inj} injection(s)")
    for i in range(n_inj):
        print(f"  Injection {i+1}: SOE = {soe[i]:.1f} deg, DOE = {doe[i]:.0f} µs")
    simout = acados_simulation(ca, x0, u, par_sim, par_model)

    p_cyl = simout['x'][0, :]
    print(f"  Peak pressure: {np.max(p_cyl)/1e5:.1f} bar "
          f"at CA = {ca[np.argmax(p_cyl)]:.1f} deg")
    print(f"  Final IMEP:    {simout['x'][2, -1]/1e5:.2f} bar")

    return simout, par_op, soe, doe


def simulate_and_plot(**kwargs):
    """Convenience wrapper: simulate then plot."""
    simout, par_op, soe, doe = simulate(**kwargs)
    fig = plot_results(simout, par_op, soe, doe)
    return simout, fig


if __name__ == '__main__':
    simulate_and_plot(soe=[-15, -5], doe=[180, 450])
