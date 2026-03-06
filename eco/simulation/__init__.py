"""Simulation module

This module contains functions for simulating the engine combustion process.
"""

from .acados_simulation import acados_simulation
from .par_op_def import par_op_def

__all__ = [
    'acados_simulation',
    'par_op_def',
]
