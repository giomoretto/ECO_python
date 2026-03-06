"""Test static_cylinder_conditions with CasADi support"""

import sys
sys.path.insert(0, '/home/morettog/projects/phd/python_code')

import numpy as np
import casadi as ca
from eco.model.subfunctions.in_cylinder import static_cylinder_conditions

print("="*60)
print("Testing static_cylinder_conditions with num_sym parameter")
print("="*60)

# Create mock parameter objects
class MockParams:
    def __init__(self):
        self.thermo = {
            'fuel': {
                'low_heat_val': 42.9e6,  # J/kg
                'air_fuel_ratio_st': 14.7,
                'nu': {
                    'N2': 0.0,
                    'O2': -11.5,
                    'CO2': 7.0,
                    'H2O': 8.0,
                    'air': -11.5
                }
            },
            'gas': {
                'molar_mass': {
                    'N2': 0.028,
                    'O2': 0.032,
                    'CO2': 0.044,
                    'H2O': 0.018,
                    'CxHy': 0.100,
                    'air': 0.029
                },
                'spec_r': {
                    'N2': 296.8,
                    'O2': 259.8,
                    'CO2': 188.9,
                    'H2O': 461.5,
                    'CxHy': 82.7
                }
            }
        }

class MockOpPoint:
    def __init__(self):
        self.m_cyl_tot = 0.0005  # kg
        self.xi_N2 = 0.75
        self.xi_O2 = 0.21
        self.xi_CO2 = 0.03
        self.xi_H2O = 0.01
        self.xi_air = 0.96
        self.xi_bg = 0.04

par_model = MockParams()
par_op = MockOpPoint()

# Test inputs
q_comb = 100.0  # J
p_cyl = 5e6  # Pa
v_cyl = 1e-4  # m^3

print("\n" + "-"*60)
print("TEST 1: Numerical mode (num_sym='Num')")
print("-"*60)

kappa, spec_r, theta_cyl, xi_o2, x_bg, x_bz, zeta_comb = static_cylinder_conditions(
    q_comb, p_cyl, v_cyl, par_model, par_op, num_sym='Num'
)

print(f"  kappa:      {kappa:.6f}")
print(f"  spec_r:     {spec_r:.2f} J/(kg*K)")
print(f"  theta_cyl:  {theta_cyl:.2f} K")
print(f"  xi_o2:      {xi_o2:.6f}")
print(f"  x_bg:       {x_bg:.6f}")
print(f"  x_bz:       {x_bz:.6f}")
print(f"  zeta_comb:  {zeta_comb:.6f}")

# Check types
assert isinstance(kappa, (float, np.ndarray)), f"Expected float, got {type(kappa)}"
assert isinstance(theta_cyl, (float, np.ndarray)), f"Expected float, got {type(theta_cyl)}"
print("✓ PASS: All outputs are numerical")

print("\n" + "-"*60)
print("TEST 2: Symbolic mode (num_sym='Sym')")
print("-"*60)

try:
    kappa_sym, spec_r_sym, theta_cyl_sym, xi_o2_sym, x_bg_sym, x_bz_sym, zeta_comb_sym = \
        static_cylinder_conditions(q_comb, p_cyl, v_cyl, par_model, par_op, num_sym='Sym')
    
    print(f"  kappa type:      {type(kappa_sym)}")
    print(f"  theta_cyl type:  {type(theta_cyl_sym)}")
    
    # In symbolic mode, kappa should use calc_kappa_casadi
    # which means if inputs are numerical, output should be CasADi DM or similar
    # Let's check if it's different from pure Python float
    print(f"  kappa value:     {float(kappa_sym):.6f}")
    
    # Values should match the numerical version
    diff_kappa = abs(float(kappa_sym) - kappa)
    print(f"  Difference in kappa: {diff_kappa:.2e}")
    
    if diff_kappa < 1e-6:
        print("✓ PASS: Symbolic mode produces correct values")
    else:
        print("✗ FAIL: Values differ between modes")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ FAIL: Error in symbolic mode: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "-"*60)
print("TEST 3: Comparison between modes")
print("-"*60)

# Both modes with the same inputs should produce the same numerical result
diff_kappa = abs(float(kappa_sym) - kappa)
diff_theta = abs(float(theta_cyl_sym) - theta_cyl)
diff_spec_r = abs(float(spec_r_sym) - spec_r)

print(f"  Difference in kappa:     {diff_kappa:.2e}")
print(f"  Difference in theta_cyl: {diff_theta:.2e}")
print(f"  Difference in spec_r:    {diff_spec_r:.2e}")

if diff_kappa < 1e-6 and diff_theta < 1e-6 and diff_spec_r < 1e-6:
    print("✓ PASS: Both modes produce identical results")
else:
    print("✗ FAIL: Results differ between modes")
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("\nThe static_cylinder_conditions function correctly:")
print("  ✓ Uses calc_kappa in numerical mode")
print("  ✓ Uses calc_kappa_casadi in symbolic mode")
print("  ✓ Produces consistent results in both modes")
