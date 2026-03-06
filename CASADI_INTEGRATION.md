# CasADi Integration for calc_kappa Function

## Summary

Successfully transcribed the MATLAB function `calcKappa_casADi.m` to Python and integrated it with the existing engine model code.

## Changes Made

### 1. Added `calc_kappa_casadi` function
**File:** `python_code/eco/model/subfunctions/in_cylinder.py`

- Transcribed the CasADi version of the kappa calculation function
- Uses CasADi's `if_else()` for symbolic conditional operations
- Supports both numerical and symbolic (CasADi SX/MX) inputs
- Maintains identical polynomial coefficients from GRI-Mech 3.0 database
- Properly handles temperature-dependent coefficient selection (theta > 1000 vs <= 1000)

### 2. Updated `static_cylinder_conditions` function
**File:** `python_code/eco/model/subfunctions/in_cylinder.py`

- Added logic to select between `calc_kappa` and `calc_kappa_casadi` based on `num_sym` parameter
- When `num_sym='Sym'`: uses `calc_kappa_casadi` (for symbolic/optimization work)
- When `num_sym='Num'`: uses `calc_kappa` (for numerical simulation)

### 3. Updated module exports
**File:** `python_code/eco/model/subfunctions/__init__.py`

- Added `calc_kappa_casadi` to the module exports
- Now available for use in other parts of the codebase

### 4. Added comprehensive tests
**Files:** 
- `python_code/tests/test_calc_kappa_casadi.py` (pytest version)
- `python_code/tests/test_calc_kappa_casadi_simple.py` (standalone version)
- `python_code/tests/test_static_cylinder_conditions_casadi.py`

Tests verify:
- ✓ Numerical evaluation matches original `calc_kappa`
- ✓ Symbolic variables are properly supported
- ✓ Gradients can be computed (essential for optimization)
- ✓ Integration with `static_cylinder_conditions` works correctly

## Key Differences: CasADi vs NumPy Version

| Aspect | calc_kappa (NumPy) | calc_kappa_casadi (CasADi) |
|--------|-------------------|---------------------------|
| **Conditional** | `if theta > 1000:` | `ca.if_else(theta > 1000, ...)` |
| **Input types** | Numerical only | Numerical or symbolic |
| **Output type** | `float` | `float`, `ca.SX`, `ca.MX`, or `ca.DM` |
| **Differentiable** | No | Yes (automatic differentiation) |
| **Use case** | Simulation | Optimization, symbolic modeling |

## Implementation Details

### CasADi's `if_else()` Function

In MATLAB:
```matlab
aK = if_else(theta > 1000, coef.(species{k})(1:5), coef.(species{k})(8:12));
```

In Python:
```python
a_k = ca.vertcat(*[
    ca.if_else(theta > 1000, coef_high[i], coef_low[i])
    for i in range(5)
])
```

The CasADi version:
- Creates a symbolic expression that can be differentiated
- Evaluates to the correct branch at runtime
- Essential for gradient-based optimization solvers

### Usage Example

```python
from eco.model.subfunctions.in_cylinder import calc_kappa_casadi
import casadi as ca

# Symbolic temperature
theta_sym = ca.SX.sym('theta')

# Numerical mass fractions
xi = {'N2': 0.75, 'O2': 0.21, 'CO2': 0.03, 'H2O': 0.01, 'CxHy': 0.0}
spec_r = {'N2': 296.8, 'O2': 259.8, 'CO2': 188.9, 'H2O': 461.5, 'CxHy': 82.7}

# Calculate symbolic kappa
kappa_sym = calc_kappa_casadi(xi, theta_sym, spec_r)

# Compute gradient
grad = ca.gradient(kappa_sym, theta_sym)
```

## Testing Results

All tests pass successfully:

```
TEST 1: Numerical values (high temperature)
  calc_kappa (NumPy):  1.30242768
  calc_kappa_casadi:   1.30242768
  ✓ PASS: Values match within tolerance

TEST 2: Numerical values (low temperature)
  calc_kappa (NumPy):  1.34586470
  calc_kappa_casadi:   1.34586470
  ✓ PASS: Values match within tolerance

TEST 3: Symbolic variables
  ✓ PASS: Result is symbolic

TEST 4: Gradient computation
  Gradient d(kappa)/d(theta): -0.0000351998
  ✓ PASS: Gradient is reasonable
```

## Benefits

1. **Optimization Ready**: The model can now be used with gradient-based optimizers like acados
2. **Automatic Differentiation**: No need to manually compute derivatives
3. **Backward Compatible**: Original numerical version remains unchanged
4. **Flexible**: Can switch between modes using `num_sym` parameter
5. **Verified**: Comprehensive tests ensure correctness

## Next Steps for Full Symbolic Model

To fully leverage this for acados optimization, you would need to:

1. Extend symbolic support to other functions in the model
2. Create fully symbolic versions of the ODE right-hand side
3. Build CasADi function objects for the complete model
4. Configure acados with the symbolic model

The `calc_kappa_casadi` function is the first step in this direction and demonstrates the pattern for converting other functions to support symbolic computation.
