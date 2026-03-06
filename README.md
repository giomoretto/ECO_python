# ECO - Engine Combustion Optimization (Python)

Python transcription of the MATLAB ECO (Engine Combustion Optimization) package for diesel engine combustion modeling and optimal control.

## Package Structure

```
eco/
├── __init__.py
├── model/                      # Physical models (01_model from MATLAB)
│   ├── __init__.py
│   ├── model_parameters.py     # Engine and thermodynamic parameters
│   ├── complete_model.py       # Complete model with combustion & NOx
│   ├── in_cylinder_model.py    # In-cylinder thermodynamics
│   ├── combustion_model.py     # Chmela combustion model
│   ├── algebraic_injector_model.py  # Fuel injection model
│   ├── ign_del_model.py        # Ignition delay model
│   ├── ignition_delay_joerg.py # Joerg ignition delay correlation
│   ├── utils.py                # Utility functions (validity checks)
│   └── subfunctions/           # Helper functions
│       ├── __init__.py
│       ├── combustion.py       # Combustion weighting functions
│       ├── ignition_delay.py   # Saturation functions
│       ├── in_cylinder.py      # Cylinder volume, kappa, heat transfer
│       └── nox.py              # Two-zone and NOx models
│
├── simulation/                 # Simulation functions (02_simulation from MATLAB)
│   ├── __init__.py
│   ├── par_op_def.py          # Operating point definition
│   ├── complete_simulation.py  # Complete combustion simulation
│   └── in_cylinder_simulation.py  # In-cylinder simulation
│
└── formulation/                # Optimal control formulation (03_formulation from MATLAB)
    ├── __init__.py
    ├── scale_unscale.py       # Variable scaling/unscaling utilities
    ├── create_acados_ocp.py   # Acados OCP problem formulation
    ├── init_acados_ocp.py     # OCP initialization with trajectories
    └── run_sqp_acados.py      # SQP solver execution
```

## Installation

The package uses a virtual environment located in `python_code/env/`.

```bash
cd python_code
source env/bin/activate  # On Linux/Mac
# OR
env\Scripts\activate  # On Windows
```

## Dependencies

- numpy >= 2.4.0
- scipy >= 1.16.3
- matplotlib >= 3.10.8
- casadi >= 3.7.2
- acados_template >= 0.5.1

## Usage

### Basic Model Usage

```python
from eco.model import ModelParameters, complete_model
from eco.simulation import par_op_def
import numpy as np

# Initialize parameters
par_model = ModelParameters()
par_op = par_op_def(par_model)

# Define state and input
x = np.array([par_op.p_int, 0, 0])  # [pressure, heat release, IMEP]
u = np.array([-5, 200])  # [SOE [degCA], DOE [µs]]
ca = -10  # Crank angle [degCA]

# Evaluate model
xdot, y = complete_model(x, u, ca, par_model, par_op)
```

### Running a Simulation

```python
from eco.simulation import complete_simulation

# Setup simulation parameters
class ParSim:
    def __init__(self):
        self.op = par_op
        self.opts = {
            'delta_phi': 0.5,  # Crank angle step [degCA]
            'int': 'RK4',      # Integration method
            'n_int': 1         # Sub-steps per crank angle step
        }

par_sim = ParSim()

# Define crank angle vector
ca = np.arange(-172, 155, 0.5)  # From IVC to EVO

# Initial conditions
x0 = np.array([par_op.p_int, 0, 0])

# Input trajectory
u = np.tile(np.array([-5, 200]), (len(ca), 1)).T

# Run simulation
simout = complete_simulation(ca, x0, u, par_sim, par_model)

# Results available in simout dict:
# - simout['x']: States over time
# - simout['y']: Outputs over time
# - simout['ca']: Crank angle vector
```

## Testing

Run tests for each module:

```bash
# Test model module
python tests/test_model.py

# Test simulation module
python tests/test_simulation.py

# Test formulation module
python tests/test_formulation.py
```

## Module Descriptions

### Model Module (`eco.model`)

Contains the physical models for engine combustion:

- **ModelParameters**: Engine geometry, thermodynamic properties, model parameters
- **complete_model**: Full model with combustion, injection, ignition delay, and optional NOx
- **in_cylinder_model**: Basic in-cylinder thermodynamics
- **combustion_model**: Chmela-based heat release rate model
- **algebraic_injector_model**: Fuel injection mass trajectory
- **ign_del_model**: Total ignition delay (chemical + physical)

### Simulation Module (`eco.simulation`)

Simulation and integration routines:

- **par_op_def**: Operating point definition and initialization
- **complete_simulation**: Full engine cycle simulation with combustion
- **in_cylinder_simulation**: Simulation with prescribed heat release

Supports multiple integration methods:
- Euler Forward (`'EulerFW'`)
- Runge-Kutta 4th order (`'RK4'`)

### Formulation Module (`eco.formulation`)

Optimal control problem formulation using acados:

- **scale_unscale**: Variable scaling/unscaling for numerical conditioning
- **create_acados_functions_inj_opt**: Acados OCP problem setup
- **create_init_acados_ocp_inj_opt**: OCP initialization with simulation trajectories
- **run_sqp_acados_inj_opt**: Execute SQP optimization and extract results

**Note**: Full implementation requires symbolic CasADi model (currently incomplete).

## Examples

See the `examples/` directory for usage demonstrations:

```bash
# Run injection optimization example
python examples/example_injection_optimization.py
```

## Key Differences from MATLAB

1. **Naming**: Python uses snake_case instead of camelCase
2. **Data structures**: Dictionaries instead of MATLAB structs
3. **Arrays**: NumPy arrays instead of MATLAB matrices
4. **Classes**: Python classes for parameters instead of MATLAB structures
5. **Type hints**: Added for better code documentation

## Status

- ✅ Model module (01_model): Complete and tested (7 tests passing)
- ✅ Simulation module (02_simulation): Complete (2 tests passing, numerical stability issue noted)
- ✅ Formulation module (03_formulation): Structure complete (5 tests passing, symbolic model incomplete)
- ✅ Example scripts: Created
- ⏳ Symbolic CasADi model: Needs implementation for full acados integration

## Known Issues

1. **Simulation numerical stability**: Complete simulation produces NaN values in some conditions - requires debugging
2. **Acados symbolic model**: The `complete_model` function needs to be rewritten to work with CasADi symbolic variables for full optimal control functionality
3. **RuntimeWarning in ignition delay**: Some invalid values in scalar power operation

## Notes

- The NOx model is included but requires enabling via `par_model.n_states = 5`
- The acados-based optimization requires symbolic models which are partially implemented
- For standalone simulation without optimization, the model and simulation modules are fully functional

## References

Based on the MATLAB ECO package for diesel engine combustion optimization.
