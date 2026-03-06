"""Simple test for calc_kappa_casadi function (no pytest)"""

import numpy as np
import sys
sys.path.insert(0, '/home/morettog/projects/phd/python_code')

try:
    import casadi as ca
    CASADI_AVAILABLE = True
    print(f"✓ CasADi version: {ca.__version__}")
except ImportError:
    CASADI_AVAILABLE = False
    print("✗ CasADi not available")
    sys.exit(1)

from eco.model.subfunctions.in_cylinder import calc_kappa, calc_kappa_casadi

print("\n" + "="*60)
print("TEST 1: Numerical values (high temperature)")
print("="*60)

# Define test inputs
xi = {
    'N2': 0.75,
    'O2': 0.21,
    'CO2': 0.03,
    'H2O': 0.01,
    'CxHy': 0.0
}

spec_r = {
    'N2': 296.8,
    'O2': 259.8,
    'CO2': 188.9,
    'H2O': 461.5,
    'CxHy': 82.7
}

# Test at high temperature (theta > 1000)
theta_high = 1500.0
kappa_casadi_high = calc_kappa_casadi(xi, theta_high, spec_r)
kappa_numpy_high = calc_kappa(xi, theta_high, spec_r)

# Convert CasADi result to float
if isinstance(kappa_casadi_high, (ca.SX, ca.MX, ca.DM)):
    kappa_casadi_high = float(kappa_casadi_high)

print(f"Temperature: {theta_high}K")
print(f"  calc_kappa (NumPy):  {kappa_numpy_high:.8f}")
print(f"  calc_kappa_casadi:   {kappa_casadi_high:.8f}")
print(f"  Difference:          {abs(kappa_casadi_high - kappa_numpy_high):.2e}")

if abs(kappa_casadi_high - kappa_numpy_high) < 1e-6:
    print("✓ PASS: Values match within tolerance")
else:
    print("✗ FAIL: Values differ too much")
    sys.exit(1)

print("\n" + "="*60)
print("TEST 2: Numerical values (low temperature)")
print("="*60)

# Test at low temperature (theta <= 1000)
theta_low = 800.0
kappa_casadi_low = calc_kappa_casadi(xi, theta_low, spec_r)
kappa_numpy_low = calc_kappa(xi, theta_low, spec_r)

# Convert CasADi result to float
if isinstance(kappa_casadi_low, (ca.SX, ca.MX, ca.DM)):
    kappa_casadi_low = float(kappa_casadi_low)

print(f"Temperature: {theta_low}K")
print(f"  calc_kappa (NumPy):  {kappa_numpy_low:.8f}")
print(f"  calc_kappa_casadi:   {kappa_casadi_low:.8f}")
print(f"  Difference:          {abs(kappa_casadi_low - kappa_numpy_low):.2e}")

if abs(kappa_casadi_low - kappa_numpy_low) < 1e-6:
    print("✓ PASS: Values match within tolerance")
else:
    print("✗ FAIL: Values differ too much")
    sys.exit(1)

print("\n" + "="*60)
print("TEST 3: Symbolic variables")
print("="*60)

# Define symbolic temperature
theta_sym = ca.SX.sym('theta')

# Define symbolic mass fractions
xi_sym = {
    'N2': ca.SX.sym('xi_N2'),
    'O2': ca.SX.sym('xi_O2'),
    'CO2': ca.SX.sym('xi_CO2'),
    'H2O': ca.SX.sym('xi_H2O'),
    'CxHy': ca.SX.sym('xi_CxHy')
}

# Calculate kappa symbolically
kappa_sym = calc_kappa_casadi(xi_sym, theta_sym, spec_r)

print(f"Result type: {type(kappa_sym)}")
print(f"Result is symbolic: {isinstance(kappa_sym, (ca.SX, ca.MX))}")

if isinstance(kappa_sym, (ca.SX, ca.MX)):
    print("✓ PASS: Result is symbolic")
else:
    print("✗ FAIL: Expected symbolic result")
    sys.exit(1)

# Create a CasADi function
kappa_func = ca.Function('kappa', 
                        [theta_sym, xi_sym['N2'], xi_sym['O2'], 
                         xi_sym['CO2'], xi_sym['H2O'], xi_sym['CxHy']],
                        [kappa_sym])

# Evaluate at a test point
theta_test = 1500.0
xi_test = [0.75, 0.21, 0.03, 0.01, 0.0]
kappa_result = float(kappa_func(theta_test, *xi_test))

# Compare with numerical version
kappa_numpy = calc_kappa(xi, theta_test, spec_r)

print(f"\nSymbolic function evaluation at {theta_test}K:")
print(f"  Symbolic function:   {kappa_result:.8f}")
print(f"  calc_kappa (NumPy):  {kappa_numpy:.8f}")
print(f"  Difference:          {abs(kappa_result - kappa_numpy):.2e}")

if abs(kappa_result - kappa_numpy) < 1e-6:
    print("✓ PASS: Symbolic evaluation matches NumPy")
else:
    print("✗ FAIL: Values differ too much")
    sys.exit(1)

print("\n" + "="*60)
print("TEST 4: Gradient computation")
print("="*60)

# Define symbolic temperature
theta_sym = ca.SX.sym('theta')

# Calculate kappa symbolically with numerical xi
kappa_sym = calc_kappa_casadi(xi, theta_sym, spec_r)

# Compute gradient
grad_kappa = ca.gradient(kappa_sym, theta_sym)

# Create function for gradient
grad_func = ca.Function('grad_kappa', [theta_sym], [grad_kappa])

# Evaluate gradient at test point
theta_test = 1500.0
grad_value = float(grad_func(theta_test))

print(f"Temperature: {theta_test}K")
print(f"  Gradient d(kappa)/d(theta): {grad_value:.10f}")

if abs(grad_value) > 1e-10 and abs(grad_value) < 1.0:
    print("✓ PASS: Gradient is reasonable")
else:
    print("✗ FAIL: Gradient is unreasonable")
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("\nThe calc_kappa_casadi function is working correctly:")
print("  ✓ Numerical evaluation matches calc_kappa")
print("  ✓ Symbolic variables are supported")
print("  ✓ Gradients can be computed")
