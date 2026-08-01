"""Complete simulation wrapper.

Re-exports acados_simulation as complete_simulation for backward compatibility
with modules that reference the original MATLAB-style name.
"""

from eco.simulation.acados_simulation import acados_simulation


def complete_simulation(ca_vec, x0, u, par_sim, par_model):
    """Run full engine simulation — wraps acados_simulation."""
    return acados_simulation(ca_vec, x0, u, par_sim, par_model)
