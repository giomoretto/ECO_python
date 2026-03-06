"""Tests for eco.model_casadi module.

All model functions use CasADi symbolic types (SX). Tests build CasADi
Functions, evaluate them numerically, and check physical plausibility
and mathematical properties.
"""

import numpy as np
import casadi as ca
import pytest

from eco.model_casadi.model_parameters import ModelParameters
from eco.model_casadi.algebraic_injector_model import algebraic_injector_model
from eco.model_casadi.ignition_delay_joerg import ignition_delay_joerg
from eco.model_casadi.ign_del_model import ign_del_model
from eco.model_casadi.combustion_model import combustion_model
from eco.model_casadi.in_cylinder_model import in_cylinder_model
from eco.model_casadi.complete_model import complete_model
from eco.model_casadi.utils import check_validity
from eco.model_casadi.subfunctions.in_cylinder import (
    cyl_vol,
    calc_kappa,
    static_cylinder_conditions,
    derive_wall_heat_transfer,
    dynamic_cylinder_conditions,
)
from eco.model_casadi.subfunctions.combustion import eval_comb_weighting_fun
from eco.model_casadi.subfunctions.ignition_delay import saturate_input
from eco.model_casadi.subfunctions.nox import (
    two_zone_model,
    _nox_eq_mole_fractions,
    nox_model,
)
from eco.simulation.par_op_def import OperatingPoint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def par_model():
    """Default ModelParameters instance."""
    return ModelParameters()


@pytest.fixture
def par_op(par_model):
    """Default OperatingPoint instance."""
    return OperatingPoint(par_model)


# ===========================================================================
# 1. ModelParameters
# ===========================================================================

class TestModelParameters:
    """Tests for ModelParameters initialisation."""

    def test_default_construction(self, par_model):
        assert par_model.eng['epsilon'] == 16.5
        assert par_model.eng['num_cyl'] == 4
        assert par_model.n_states == 3

    def test_engine_geometry_consistency(self, par_model):
        """vol_dis = vol_clear * (epsilon - 1)"""
        expected = par_model.eng['vol_clear'] * (par_model.eng['epsilon'] - 1)
        assert par_model.eng['vol_dis'] == pytest.approx(expected, rel=1e-10)

    def test_stoichiometric_coefficients_sign(self, par_model):
        """O2 and fuel stoich. coeff should be negative (consumed)."""
        assert par_model.thermo['fuel']['nu']['O2'] < 0
        assert par_model.thermo['fuel']['nu']['CxHy'] == -1

    def test_species_mass_fractions_sum(self, par_model):
        """Air mass fractions O2 + N2 must equal 1."""
        total = (par_model.thermo['xi']['air']['O2'] +
                 par_model.thermo['xi']['air']['N2'])
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_burnt_gas_fractions_sum(self, par_model):
        """Burnt-gas mole fractions should sum to 1."""
        psi = par_model.thermo['psi']['bg']
        total = psi['H2O'] + psi['CO2'] + psi['N2']
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_load_parameters(self):
        x_sol = np.array([1e-4, 2000, 0, 3500, 3e-4, 2.5e-4, 3e-4, 0, 2.5e-3])
        par = ModelParameters(x_sol)
        assert par.comb['c_dif']['a'] == x_sol[0]
        assert par.ign['phys']['const'] == x_sol[4]
        assert par.ign['tau_max'] == x_sol[8]


# ===========================================================================
# 2. Subfunctions – ignition_delay.py (saturate_input)
# ===========================================================================

class TestSaturateInput:
    """Tests for smooth saturation function."""

    def test_within_bounds_unchanged(self):
        """Value well inside bounds should pass through ~unchanged."""
        u = ca.SX.sym('u')
        y = saturate_input(u, 0, 10)
        f = ca.Function('f', [u], [y])
        assert float(f(5)) == pytest.approx(5.0, abs=0.05)

    def test_clips_below_min(self):
        u = ca.SX.sym('u')
        y = saturate_input(u, 2, 8)
        f = ca.Function('f', [u], [y])
        assert float(f(-10)) == pytest.approx(2.0, abs=0.1)

    def test_clips_above_max(self):
        u = ca.SX.sym('u')
        y = saturate_input(u, 2, 8)
        f = ca.Function('f', [u], [y])
        assert float(f(100)) == pytest.approx(8.0, abs=0.1)

    def test_monotonicity(self):
        """Output should be monotonically non-decreasing."""
        u = ca.SX.sym('u')
        y = saturate_input(u, 0, 10)
        f = ca.Function('f', [u], [y])
        vals = [float(f(v)) for v in np.linspace(-5, 15, 50)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))

    def test_differentiable(self):
        """Gradient must exist everywhere (smooth saturation)."""
        u = ca.SX.sym('u')
        y = saturate_input(u, 0, 10)
        dy = ca.jacobian(y, u)
        f = ca.Function('f', [u], [dy])
        for v in [-5, 0, 5, 10, 15]:
            g = float(f(v))
            assert np.isfinite(g)
            assert g >= 0  # non-negative gradient


# ===========================================================================
# 3. Subfunctions – combustion.py (eval_comb_weighting_fun)
# ===========================================================================

class TestCombustionWeighting:
    """Tests for premixed/diffusive weighting sigmoid."""

    def test_at_midpoint(self):
        tau = ca.SX.sym('tau')
        gamma = eval_comb_weighting_fun(tau, 1e-4, 3e-4)
        f = ca.Function('f', [tau], [gamma])
        assert float(f(2e-4)) == pytest.approx(0.5, abs=0.01)

    def test_limits(self):
        tau = ca.SX.sym('tau')
        gamma = eval_comb_weighting_fun(tau, 1e-4, 3e-4)
        f = ca.Function('f', [tau], [gamma])
        # Well below tau_min -> 0, well above tau_max -> 1
        assert float(f(-1e-3)) < 0.01
        assert float(f(1e-2)) > 0.99

    def test_output_range(self):
        tau = ca.SX.sym('tau')
        gamma = eval_comb_weighting_fun(tau, 1e-4, 3e-4)
        f = ca.Function('f', [tau], [gamma])
        for v in np.linspace(-1e-3, 1e-2, 30):
            g = float(f(v))
            assert 0 <= g <= 1

    def test_monotonically_increasing(self):
        tau = ca.SX.sym('tau')
        gamma = eval_comb_weighting_fun(tau, 1e-4, 3e-4)
        f = ca.Function('f', [tau], [gamma])
        vals = [float(f(v)) for v in np.linspace(-1e-3, 1e-2, 50)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


# ===========================================================================
# 4. Subfunctions – in_cylinder.py
# ===========================================================================

class TestCylVol:
    """Tests for cylinder volume calculation."""

    def test_tdc_is_minimum(self, par_model):
        """Volume at TDC (0 deg) should equal clearance volume."""
        ca_sym = ca.SX.sym('ca')
        v, dv, s = cyl_vol(ca_sym, par_model.eng)
        f = ca.Function('f', [ca_sym], [v])
        v_tdc = float(f(0))
        assert v_tdc == pytest.approx(par_model.eng['vol_clear'], rel=0.01)

    def test_bdc_is_maximum(self, par_model):
        """Volume at BDC (180 deg) should be vol_clear + vol_dis."""
        ca_sym = ca.SX.sym('ca')
        v, dv, s = cyl_vol(ca_sym, par_model.eng)
        f = ca.Function('f', [ca_sym], [v])
        v_bdc = float(f(180))
        expected = par_model.eng['vol_clear'] + par_model.eng['vol_dis']
        assert v_bdc == pytest.approx(expected, rel=0.01)

    def test_volume_derivative_zero_at_tdc(self, par_model):
        """dV/d(phi) should be ~0 at TDC."""
        ca_sym = ca.SX.sym('ca')
        v, dv, s = cyl_vol(ca_sym, par_model.eng)
        f = ca.Function('f', [ca_sym], [dv])
        assert float(f(0)) == pytest.approx(0, abs=1e-10)

    def test_volume_derivative_zero_at_bdc(self, par_model):
        """dV/d(phi) should be ~0 at BDC."""
        ca_sym = ca.SX.sym('ca')
        v, dv, s = cyl_vol(ca_sym, par_model.eng)
        f = ca.Function('f', [ca_sym], [dv])
        assert float(f(180)) == pytest.approx(0, abs=1e-10)

    def test_volume_positive_everywhere(self, par_model):
        """Volume must be positive for all crank angles."""
        ca_sym = ca.SX.sym('ca')
        v, dv, s = cyl_vol(ca_sym, par_model.eng)
        f = ca.Function('f', [ca_sym], [v])
        for deg in np.linspace(-180, 180, 100):
            assert float(f(deg)) > 0

    def test_symmetry(self, par_model):
        """Volume should be symmetric about TDC."""
        ca_sym = ca.SX.sym('ca')
        v, dv, s = cyl_vol(ca_sym, par_model.eng)
        f = ca.Function('f', [ca_sym], [v])
        for deg in [30, 60, 90, 120, 150]:
            assert float(f(deg)) == pytest.approx(float(f(-deg)), rel=1e-10)


class TestCalcKappa:
    """Tests for heat capacity ratio calculation."""

    def test_reasonable_range(self, par_model):
        """Kappa for air-like mixture should be between 1.2 and 1.5."""
        theta = ca.SX.sym('theta')
        xi = {
            'N2': 0.76,
            'O2': 0.23,
            'CO2': 0.005,
            'H2O': 0.005,
            'CxHy': 0.0,
        }
        kappa = calc_kappa(xi, theta, par_model.thermo['gas']['spec_r'])
        f = ca.Function('f', [theta], [kappa])
        for T in [300, 500, 1000, 1500, 2000]:
            k = float(f(T))
            assert 1.2 < k < 1.5, f"kappa={k} at T={T}K"

    def test_decreases_with_temperature(self, par_model):
        """Kappa generally decreases as temperature rises."""
        theta = ca.SX.sym('theta')
        xi = {'N2': 0.76, 'O2': 0.23, 'CO2': 0.005, 'H2O': 0.005, 'CxHy': 0.0}
        kappa = calc_kappa(xi, theta, par_model.thermo['gas']['spec_r'])
        f = ca.Function('f', [theta], [kappa])
        k_low = float(f(400))
        k_high = float(f(2000))
        assert k_low > k_high


class TestStaticCylinderConditions:
    """Tests for static_cylinder_conditions."""

    def test_zero_combustion(self, par_model, par_op):
        """With q_comb=0, temperature should follow ideal gas law at IVC."""
        q = ca.SX.sym('q')
        p = ca.SX.sym('p')
        v = ca.SX.sym('v')
        kappa, spec_r, theta, xi_o2, x_bg, x_bz, zeta = \
            static_cylinder_conditions(q, p, v, par_model, par_op)
        f = ca.Function('f', [q, p, v], [theta, xi_o2, x_bz, zeta])
        out = f(0, par_op.p_int, float(par_op.v_int))
        theta_val = float(out[0])
        # Should be close to IVC temperature (ideal gas)
        assert theta_val == pytest.approx(float(par_op.theta_ivc), rel=0.01)

    def test_oxygen_decreases_with_combustion(self, par_model, par_op):
        """Oxygen fraction should decrease as more fuel is burned."""
        q = ca.SX.sym('q')
        p = ca.SX.sym('p')
        v = ca.SX.sym('v')
        _, _, _, xi_o2, _, _, _ = \
            static_cylinder_conditions(q, p, v, par_model, par_op)
        f = ca.Function('f', [q, p, v], [xi_o2])
        xi_0 = float(f(0, 1e6, 1e-4))
        xi_q = float(f(100, 1e6, 1e-4))
        assert xi_q < xi_0

    def test_burned_zone_fraction_increases(self, par_model, par_op):
        """x_bz should increase with combustion heat release."""
        q = ca.SX.sym('q')
        p = ca.SX.sym('p')
        v = ca.SX.sym('v')
        _, _, _, _, _, x_bz, _ = \
            static_cylinder_conditions(q, p, v, par_model, par_op)
        f = ca.Function('f', [q, p, v], [x_bz])
        xbz_0 = float(f(0, 1e6, 1e-4))
        xbz_q = float(f(200, 1e6, 1e-4))
        assert xbz_q > xbz_0


class TestWallHeatTransfer:
    """Tests for derive_wall_heat_transfer."""

    def test_heat_loss_is_negative(self, par_model):
        """Wall heat transfer should cool the gas (negative when T_cyl > T_wall)."""
        ca_sym = ca.SX.sym('ca')
        stroke = ca.SX.sym('stroke')
        theta = ca.SX.sym('theta')
        p = ca.SX.sym('p')
        dq = derive_wall_heat_transfer(
            ca_sym, stroke, theta, p,
            theta_ivc=350, p_ivc=1.2e5, vol_ivc=5e-4,
            vol=3e-4, eng_spd=33.3, par_model=par_model)
        f = ca.Function('f', [ca_sym, stroke, theta, p], [dq])
        # High temperature gas → heat flows out of cylinder
        val = float(f(10, 0.04, 2000, 80e5))
        assert val < 0

    def test_heat_gain_at_low_temperature(self, par_model):
        """At very low gas temperature, wall heat transfer can be positive."""
        ca_sym = ca.SX.sym('ca')
        stroke = ca.SX.sym('stroke')
        theta = ca.SX.sym('theta')
        p = ca.SX.sym('p')
        dq = derive_wall_heat_transfer(
            ca_sym, stroke, theta, p,
            theta_ivc=350, p_ivc=1.2e5, vol_ivc=5e-4,
            vol=3e-4, eng_spd=33.3, par_model=par_model)
        f = ca.Function('f', [ca_sym, stroke, theta, p], [dq])
        # Very low temperature gas → heat flows into the gas
        val = float(f(10, 0.04, 200, 1e5))
        assert val > 0


class TestDynamicCylinderConditions:
    """Tests for dynamic_cylinder_conditions."""

    def test_no_heat_release_gives_compression(self, par_model, par_op):
        """With dq_comb=0, pressure should rise during compression (dv < 0)."""
        ca_sym = ca.SX.sym('ca')
        p = ca.SX.sym('p')
        dq = ca.SX.sym('dq')
        v_cyl = ca.SX.sym('v')
        dv_cyl = ca.SX.sym('dv')
        strk = ca.SX.sym('strk')
        theta = ca.SX.sym('theta')

        # Disable wall heat loss for clean test
        par_test = ModelParameters()
        par_test.opts['enable_wall_heat_loss'] = False

        dp, dimep, dq_wall = dynamic_cylinder_conditions(
            ca_sym, p, dq, v_cyl, dv_cyl, strk, theta,
            1.35, 287.0, par_test, par_op)
        f = ca.Function('f', [ca_sym, p, dq, v_cyl, dv_cyl, strk, theta],
                        [dp, dimep, dq_wall])
        # Compression: dv < 0, dq_comb = 0
        out = f(0, 50e5, 0, 5e-5, -1e-7, 0.04, 800)
        dp_val = float(out[0])
        assert dp_val > 0, "Pressure should increase during compression"

    def test_imep_derivative_sign(self, par_model, par_op):
        """dIMEP should be positive when p_cyl * dv > 0 (expansion work)."""
        par_test = ModelParameters()
        par_test.opts['enable_wall_heat_loss'] = False
        ca_sym = ca.SX.sym('ca')
        p = ca.SX.sym('p')
        dq = ca.SX.sym('dq')
        v_cyl = ca.SX.sym('v')
        dv_cyl = ca.SX.sym('dv')
        strk = ca.SX.sym('strk')
        theta = ca.SX.sym('theta')
        dp, dimep, _ = dynamic_cylinder_conditions(
            ca_sym, p, dq, v_cyl, dv_cyl, strk, theta,
            1.35, 287.0, par_test, par_op)
        f = ca.Function('f', [ca_sym, p, dq, v_cyl, dv_cyl, strk, theta], [dimep])
        # Expansion: dv > 0, p > 0
        val = float(f(30, 50e5, 0, 1e-4, 1e-7, 0.04, 1000))
        assert val > 0


# ===========================================================================
# 5. ignition_delay_joerg
# ===========================================================================

class TestIgnitionDelayJoerg:
    """Tests for ignition_delay_joerg."""

    def test_positive_delay(self):
        """Ignition delay must be positive."""
        p = ca.SX.sym('p')
        T = ca.SX.sym('T')
        x_o2 = ca.SX.sym('x_o2')
        tau = ignition_delay_joerg(p, T, x_o2)
        f = ca.Function('f', [p, T, x_o2], [tau])
        val = float(f(80e5, 900, 0.21))
        assert val > 0

    def test_decreases_with_temperature(self):
        """Higher temperature should decrease ignition delay."""
        p = ca.SX.sym('p')
        T = ca.SX.sym('T')
        x_o2 = ca.SX.sym('x_o2')
        tau = ignition_delay_joerg(p, T, x_o2)
        f = ca.Function('f', [p, T, x_o2], [tau])
        tau_low = float(f(80e5, 700, 0.21))
        tau_high = float(f(80e5, 1100, 0.21))
        assert tau_high < tau_low

    def test_decreases_with_pressure(self):
        """Higher pressure should generally decrease ignition delay."""
        p = ca.SX.sym('p')
        T = ca.SX.sym('T')
        x_o2 = ca.SX.sym('x_o2')
        tau = ignition_delay_joerg(p, T, x_o2)
        f = ca.Function('f', [p, T, x_o2], [tau])
        tau_low_p = float(f(40e5, 900, 0.21))
        tau_high_p = float(f(120e5, 900, 0.21))
        assert tau_high_p < tau_low_p

    def test_finite_output(self):
        """Output must be finite for typical engine conditions."""
        p = ca.SX.sym('p')
        T = ca.SX.sym('T')
        x_o2 = ca.SX.sym('x_o2')
        tau = ignition_delay_joerg(p, T, x_o2)
        f = ca.Function('f', [p, T, x_o2], [tau])
        for pv in [40e5, 80e5, 120e5]:
            for Tv in [600, 800, 1000]:
                val = float(f(pv, Tv, 0.21))
                assert np.isfinite(val)


# ===========================================================================
# 6. ign_del_model
# ===========================================================================

class TestIgnDelModel:
    """Tests for total ignition delay model."""

    def test_bounded_by_saturation(self, par_model):
        """Output should be bounded between tau_min and tau_max."""
        p = ca.SX.sym('p')
        T = ca.SX.sym('T')
        x_o2 = ca.SX.sym('x_o2')
        tau = ign_del_model(p, T, x_o2, par_model)
        f = ca.Function('f', [p, T, x_o2], [tau])
        for pv in [30e5, 80e5, 150e5]:
            for Tv in [500, 800, 1200]:
                val = float(f(pv, Tv, 0.21))
                assert val >= par_model.ign['tau_min'] - 0.1 * abs(par_model.ign['tau_min']) - 1e-8
                assert val <= par_model.ign['tau_max'] + 0.1 * abs(par_model.ign['tau_max']) + 1e-8

    def test_includes_physical_delay(self, par_model):
        """Total delay should be at least the physical component (except saturation)."""
        p = ca.SX.sym('p')
        T = ca.SX.sym('T')
        x_o2 = ca.SX.sym('x_o2')
        tau = ign_del_model(p, T, x_o2, par_model)
        f = ca.Function('f', [p, T, x_o2], [tau])
        val = float(f(80e5, 900, 0.21))
        # Should be >= physical delay (but saturation may clip it)
        assert val >= par_model.ign['phys']['const'] * 0.5


# ===========================================================================
# 7. algebraic_injector_model
# ===========================================================================

class TestAlgebraicInjectorModel:
    """Tests for fuel injection trajectory model."""

    def _make_function(self, n_inj=1):
        t = ca.SX.sym('t')
        soe = ca.SX.sym('soe', n_inj)
        doe = ca.SX.sym('doe', n_inj)
        m_act, m_tot = algebraic_injector_model(t, 1000e5, soe, doe, n_inj)
        return ca.Function('f', [t, soe, doe], [m_act, m_tot])

    def test_no_injection_before_soe(self):
        """Before start of energising, no fuel should be injected."""
        f = self._make_function(1)
        out = f(0, 500, 300)  # t=0 well before SOE=500
        m_act = float(out[0])
        assert m_act == pytest.approx(0, abs=1e-10)

    def test_total_mass_positive(self):
        """Total accumulated mass should be positive for valid injection."""
        f = self._make_function(1)
        out = f(1500, 200, 500)  # t well after injection
        m_tot = float(out[1])
        assert m_tot > 0

    def test_mass_increases_with_doe(self):
        """Larger duration of energising should inject more fuel."""
        f = self._make_function(1)
        out1 = f(2000, 200, 200)
        out2 = f(2000, 200, 600)
        assert float(out2[1]) > float(out1[1])

    def test_two_injections(self):
        """Two injections should yield more total mass than one."""
        f1 = self._make_function(1)
        f2 = self._make_function(2)
        out1 = f1(3000, 200, 400)
        out2 = f2(3000, ca.DM([200, 1000]), ca.DM([400, 400]))
        assert float(out2[1]) > float(out1[1])

    def test_finite_outputs(self):
        """Outputs must always be finite."""
        f = self._make_function(1)
        for t_val in [0, 200, 500, 1000, 2000]:
            out = f(t_val, 300, 400)
            assert np.isfinite(float(out[0]))
            assert np.isfinite(float(out[1]))


# ===========================================================================
# 8. combustion_model
# ===========================================================================

class TestCombustionModel:
    """Tests for Chmela combustion model."""

    def test_no_fuel_no_combustion(self, par_model):
        """Zero prepared fuel should give ~zero heat release."""
        q = ca.SX.sym('q')
        mf = ca.SX.sym('mf')
        tau = ca.SX.sym('tau')
        dq = combustion_model(q, mf, tau, 33.3, par_model)
        f = ca.Function('f', [q, mf, tau], [dq])
        val = float(f(0, 0, 5e-4))
        assert val == pytest.approx(0, abs=0.1)

    def test_positive_heat_release(self, par_model):
        """With fuel available, heat release rate should be positive."""
        q = ca.SX.sym('q')
        mf = ca.SX.sym('mf')
        tau = ca.SX.sym('tau')
        dq = combustion_model(q, mf, tau, 33.3, par_model)
        f = ca.Function('f', [q, mf, tau], [dq])
        val = float(f(0, 10e-6, 5e-4))
        assert val > 0

    def test_heat_release_decreases_as_fuel_burns(self, par_model):
        """As q_comb approaches total energy, rate should drop."""
        q = ca.SX.sym('q')
        mf = ca.SX.sym('mf')
        tau = ca.SX.sym('tau')
        dq = combustion_model(q, mf, tau, 33.3, par_model)
        f = ca.Function('f', [q, mf, tau], [dq])
        m_fuel = 20e-6
        q_max = m_fuel * par_model.thermo['fuel']['low_heat_val']
        rate_early = float(f(0, m_fuel, 5e-4))
        rate_late = float(f(0.9 * q_max, m_fuel, 5e-4))
        assert rate_early > rate_late

    def test_differentiable(self, par_model):
        """Combustion model should be differentiable (smooth max)."""
        q = ca.SX.sym('q')
        mf = ca.SX.sym('mf')
        tau = ca.SX.sym('tau')
        dq = combustion_model(q, mf, tau, 33.3, par_model)
        jac = ca.jacobian(dq, ca.vertcat(q, mf, tau))
        f = ca.Function('f', [q, mf, tau], [jac])
        out = np.array(f(10, 10e-6, 5e-4)).flatten()
        assert all(np.isfinite(out))


# ===========================================================================
# 9. in_cylinder_model
# ===========================================================================

class TestInCylinderModel:
    """Tests for simplified in-cylinder model."""

    def test_output_shapes(self, par_model, par_op):
        """xdot should have 3 elements, y should have 4."""
        x = ca.SX.sym('x', 3)
        u = ca.SX.sym('u')
        phi = ca.SX.sym('phi')
        xdot, y = in_cylinder_model(x, u, phi, par_model, par_op)
        assert xdot.shape == (3, 1)
        assert y.shape == (4, 1)

    def test_motored_operation(self, par_model, par_op):
        """With zero heat release input, only pressure and volume exchange."""
        x = ca.SX.sym('x', 3)
        u = ca.SX.sym('u')
        phi = ca.SX.sym('phi')
        xdot, y = in_cylinder_model(x, u, phi, par_model, par_op)
        f = ca.Function('f', [x, u, phi], [xdot, y])
        x0 = ca.DM([par_op.p_int, 0, 0])
        out_xdot, out_y = f(x0, 0, -170)
        # dq_comb = 0 so second state derivative = 0
        assert float(out_xdot[1]) == pytest.approx(0, abs=1e-10)

    def test_finite_outputs(self, par_model, par_op):
        """All outputs must be finite for typical conditions."""
        x = ca.SX.sym('x', 3)
        u = ca.SX.sym('u')
        phi = ca.SX.sym('phi')
        xdot, y = in_cylinder_model(x, u, phi, par_model, par_op)
        f = ca.Function('f', [x, u, phi], [xdot, y])
        x0 = ca.DM([par_op.p_int, 0, 0])
        out_xdot, out_y = f(x0, 0, -170)
        for i in range(3):
            assert np.isfinite(float(out_xdot[i]))
        for i in range(4):
            assert np.isfinite(float(out_y[i]))


# ===========================================================================
# 10. utils – check_validity
# ===========================================================================

class TestCheckValidity:
    """Tests for state validity checks."""

    def test_no_nox_shape(self, par_model, par_op):
        """Without NOx, output should have 3 elements."""
        x = ca.SX.sym('x', 3)
        phi = ca.SX.sym('phi')
        x_corr = check_validity(x, phi, par_model, par_op, en_nox=False)
        assert x_corr.shape == (3, 1)

    def test_with_nox_shape(self, par_model, par_op):
        """With NOx, output should have 5 elements."""
        x = ca.SX.sym('x', 5)
        phi = ca.SX.sym('phi')
        x_corr = check_validity(x, phi, par_model, par_op, en_nox=True)
        assert x_corr.shape == (5, 1)

    def test_pressure_lower_bound(self, par_model, par_op):
        """Pressure should be enforced above motored lower bound."""
        x = ca.SX.sym('x', 3)
        phi = ca.SX.sym('phi')
        x_corr = check_validity(x, phi, par_model, par_op, en_nox=False)
        f = ca.Function('f', [x, phi], [x_corr])
        # Set pressure very low
        x0 = ca.DM([1e3, 10, 0])  # 1 kPa, very low
        out = f(x0, -10)
        p_corrected = float(out[0])
        assert p_corrected > 1e3, "Pressure should be raised by lower bound"

    def test_nox_lower_bound(self, par_model, par_op):
        """NOx should not go negative."""
        x = ca.SX.sym('x', 5)
        phi = ca.SX.sym('phi')
        x_corr = check_validity(x, phi, par_model, par_op, en_nox=True)
        f = ca.Function('f', [x, phi], [x_corr])
        # Set NOx negative
        x0 = ca.DM([50e5, 10, 0, 500, -1e-3])
        out = f(x0, 0)
        no_corrected = float(out[4])
        assert no_corrected >= -1e-9


# ===========================================================================
# 11. NOx subfunctions
# ===========================================================================

class TestNoxEquilibriumFractions:
    """Tests for _nox_eq_mole_fractions."""

    def test_positive_fractions(self):
        """All equilibrium fractions should be positive."""
        theta = ca.SX.sym('theta')
        x_n, x_o, x_n2, x_o2, x_oh, x_no = _nox_eq_mole_fractions(theta)
        f = ca.Function('f', [theta], [x_n, x_o, x_n2, x_o2, x_oh, x_no])
        for T in [1800, 2200, 2600, 3000]:
            out = f(T)
            for i in range(6):
                assert float(out[i]) > 0, f"Species {i} negative at T={T}"

    def test_no_increases_with_temperature(self):
        """NO equilibrium should increase with temperature."""
        theta = ca.SX.sym('theta')
        fracs = _nox_eq_mole_fractions(theta)
        x_no = fracs[5]
        f = ca.Function('f', [theta], [x_no])
        no_low = float(f(1800))
        no_high = float(f(2800))
        assert no_high > no_low


class TestTwoZoneModel:
    """Tests for two-zone temperature model."""

    def test_output_shapes(self, par_model, par_op):
        """Should return two scalar outputs."""
        q_comb = ca.SX.sym('q')
        theta_uz = ca.SX.sym('tuz')
        dp = ca.SX.sym('dp')
        p = ca.SX.sym('p')
        dv = ca.SX.sym('dv')
        v = ca.SX.sym('v')
        t_cyl = ca.SX.sym('tcyl')
        kappa = ca.SX.sym('kappa')
        d_tuz, theta_bz = two_zone_model(
            q_comb, theta_uz, dp, p, dv, v, t_cyl, kappa,
            par_op.m_cyl_tot, par_op, par_model, 0.1, 0.1)
        assert d_tuz.shape == (1, 1)
        assert theta_bz.shape == (1, 1)

    def test_burned_zone_hotter_than_unburned(self, par_model, par_op):
        """Burned zone temperature should be higher than unburned zone."""
        q_comb = ca.SX.sym('q')
        theta_uz = ca.SX.sym('tuz')
        dp = ca.SX.sym('dp')
        p = ca.SX.sym('p')
        dv = ca.SX.sym('dv')
        v = ca.SX.sym('v')
        t_cyl = ca.SX.sym('tcyl')
        kappa = ca.SX.sym('kappa')
        d_tuz, theta_bz = two_zone_model(
            q_comb, theta_uz, dp, p, dv, v, t_cyl, kappa,
            par_op.m_cyl_tot, par_op, par_model, 0.1, 0.2)
        f = ca.Function('f', [q_comb, theta_uz, dp, p, dv, v, t_cyl, kappa],
                        [theta_bz])
        # Significant combustion: high t_cyl, moderate unburned zone temp
        tbz = float(f(500, 600, 1e10, 80e5, 1e-7, 1e-4, 1200, 1.35))
        assert tbz > 600


# ===========================================================================
# 12. complete_model
# ===========================================================================

class TestCompleteModel:
    """Tests for the full complete_model."""

    def test_output_shapes_no_nox(self, par_model, par_op):
        """Without NOx: xdot=3, y=8."""
        x = ca.SX.sym('x', 3)
        u = ca.SX.sym('u', 2)  # 1 injection: [SOE, DOE]
        phi = ca.SX.sym('phi')
        xdot, y = complete_model(x, u, phi, par_model, par_op, en_nox=False)
        assert xdot.shape == (3, 1)
        assert y.shape == (8, 1)

    def test_output_shapes_with_nox(self, par_model, par_op):
        """With NOx: xdot=5, y=12."""
        x = ca.SX.sym('x', 5)
        u = ca.SX.sym('u', 2)
        phi = ca.SX.sym('phi')
        xdot, y = complete_model(x, u, phi, par_model, par_op, en_nox=True)
        assert xdot.shape == (5, 1)
        assert y.shape == (12, 1)

    def test_finite_no_nox(self, par_model, par_op):
        """All outputs should be finite for a typical evaluation (no NOx)."""
        x = ca.SX.sym('x', 3)
        u = ca.SX.sym('u', 2)
        phi = ca.SX.sym('phi')
        xdot, y = complete_model(x, u, phi, par_model, par_op, en_nox=False)
        f = ca.Function('f', [x, u, phi], [xdot, y])
        x0 = ca.DM([par_op.p_int, 0, 0])
        u0 = ca.DM([-10, 300])  # SOE=-10 degCA, DOE=300 µs
        out_xdot, out_y = f(x0, u0, -170)
        for i in range(3):
            assert np.isfinite(float(out_xdot[i])), f"xdot[{i}] not finite"
        for i in range(8):
            assert np.isfinite(float(out_y[i])), f"y[{i}] not finite"

    def test_finite_with_nox(self, par_model, par_op):
        """All outputs should be finite for a typical evaluation (with NOx)."""
        x = ca.SX.sym('x', 5)
        u = ca.SX.sym('u', 2)
        phi = ca.SX.sym('phi')
        xdot, y = complete_model(x, u, phi, par_model, par_op, en_nox=True)
        f = ca.Function('f', [x, u, phi], [xdot, y])
        x0 = ca.DM([par_op.p_int, 0, 0, float(par_op.theta_ivc), 1e-10])
        u0 = ca.DM([-10, 300])
        out_xdot, out_y = f(x0, u0, -170)
        for i in range(5):
            assert np.isfinite(float(out_xdot[i])), f"xdot[{i}] not finite"
        for i in range(12):
            assert np.isfinite(float(out_y[i])), f"y[{i}] not finite"

    def test_two_injections_shape(self, par_model, par_op):
        """With 2 injections: u has 4 elements, shapes still correct."""
        x = ca.SX.sym('x', 3)
        u = ca.SX.sym('u', 4)  # [SOE1, SOE2, DOE1, DOE2]
        phi = ca.SX.sym('phi')
        xdot, y = complete_model(x, u, phi, par_model, par_op, en_nox=False)
        assert xdot.shape == (3, 1)
        assert y.shape == (8, 1)

    def test_casadi_function_creation(self, par_model, par_op):
        """Should be possible to wrap in a CasADi Function and evaluate."""
        x = ca.SX.sym('x', 3)
        u = ca.SX.sym('u', 2)
        phi = ca.SX.sym('phi')
        xdot, y = complete_model(x, u, phi, par_model, par_op, en_nox=False)
        f = ca.Function('model', [x, u, phi], [xdot, y])
        # Should not raise
        assert f.n_in() == 3
        assert f.n_out() == 2

    def test_jacobian_exists(self, par_model, par_op):
        """Jacobian w.r.t. states should be computable (differentiable model)."""
        x = ca.SX.sym('x', 3)
        u = ca.SX.sym('u', 2)
        phi = ca.SX.sym('phi')
        xdot, y = complete_model(x, u, phi, par_model, par_op, en_nox=False)
        J = ca.jacobian(xdot, x)
        f = ca.Function('jacobian_fn', [x, u, phi], [J])
        x0 = ca.DM([par_op.p_int, 0, 0])
        u0 = ca.DM([-10, 300])
        J_val = np.array(f(x0, u0, -170))
        assert J_val.shape == (3, 3)
        assert np.all(np.isfinite(J_val))


# ===========================================================================
# 13. OperatingPoint
# ===========================================================================

class TestOperatingPoint:
    """Tests for OperatingPoint construction."""

    def test_default_values(self, par_model):
        op = OperatingPoint(par_model)
        assert op.eng_spd == pytest.approx(2000 / 60)
        assert op.p_im == pytest.approx(1.2e5)

    def test_mass_fractions_sum(self, par_op):
        total = par_op.xi_n2 + par_op.xi_o2 + par_op.xi_co2 + par_op.xi_h2o + par_op.xi_cxhy
        assert total == pytest.approx(1.0, abs=1e-8)

    def test_positive_cylinder_mass(self, par_op):
        assert par_op.m_cyl_tot > 0

    def test_ivc_temperature_plausible(self, par_op):
        """IVC temperature should be between ambient and ~500 K."""
        theta = float(par_op.theta_ivc)
        assert 250 < theta < 500

    def test_custom_measurements(self, par_model):
        meas = {
            'ne': 3000 / 60,
            'p_im': 2.0e5,
            't_im': 330,
            'x_bg': 0.2,
            'p_rail': 1500e5,
        }
        op = OperatingPoint(par_model, meas=meas)
        assert op.eng_spd == pytest.approx(3000 / 60)
        assert op.p_rail == pytest.approx(1500e5)
