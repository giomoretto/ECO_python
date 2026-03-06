"""Shared plotting utilities for simulation examples."""

import numpy as np
import matplotlib.pyplot as plt
from eco.model_casadi.utils import injection_pattern


def plot_results(simout, par_op, soe, doe):
    """Plot simulation results: HRR, cylinder pressure, and temperatures.

    Args:
        simout: Dictionary with keys 'ca', 'x', 'xdot', 'y'.
        par_op: Operating point parameters.
        soe: SOE array [degCA].
        doe: DOE array [µs].

    Returns:
        fig: Matplotlib figure.
    """
    ca = simout['ca']
    p_cyl = simout['x'][0, :]
    n_inj = len(soe)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    inj_str = ', '.join(f'Inj{i+1}: {soe[i]:.0f}°/{doe[i]:.0f} µs'
                        for i in range(n_inj))

    # Left: Heat release rate + injection pattern
    dq_comb = simout['xdot'][1, :]  # [J/degCA]
    ax1.plot(ca, dq_comb, linewidth=1.5, color='black', label='HRR')
    ax1.set_xlabel('Crank Angle [deg aTDC]')
    ax1.set_ylabel('Heat Release Rate [J/degCA]')
    ax1.set_title('Heat Release Rate')
    ax1.set_xlim(-20, 30)
    ax1.grid(True, alpha=0.3)

    # Injection pattern on twin axis
    ca_pat, u_pat = injection_pattern(soe, doe, par_op.eng_spd, ca_min=-20, ca_max=30)
    ax1b = ax1.twinx()
    ax1b.fill_between(ca_pat, u_pat, alpha=0.15, color='black', step='pre')
    ax1b.plot(ca_pat, u_pat, linewidth=1.0, color='black', label='Injection')
    ax1b.set_ylabel('Injection [-]')
    ax1b.set_ylim(-0.05, 5)
    ax1b.set_yticks([0, 1])

    # Middle: Cylinder pressure
    ax2.plot(ca, p_cyl / 1e5, linewidth=1.5, color='black')
    ax2.set_xlabel('Crank Angle [deg aTDC]')
    ax2.set_ylabel('Cylinder Pressure [bar]')
    ax2.set_title('Cylinder Pressure')
    ax2.set_xlim(-20, 30)
    ax2.set_ylim(20, 150)
    ax2.grid(True, alpha=0.3)

    # Right: Temperatures
    theta_cyl = simout['y'][0, :]    # mean cylinder temperature [K]
    theta_uz = simout['x'][3, :]     # unburned zone temperature [K]
    theta_bz = simout['y'][7, :]     # burned zone temperature [K]
    ax3.plot(ca, theta_cyl, linewidth=1.5, color='black', label='Cylinder (mean)')
    ax3.plot(ca, theta_uz, linewidth=1.5, color='tab:blue', label='Unburned zone')
    ax3.plot(ca, theta_bz, linewidth=1.5, color='tab:red', label='Burned zone')
    ax3.set_xlabel('Crank Angle [deg aTDC]')
    ax3.set_ylabel('Temperature [K]')
    ax3.set_title('Temperatures')
    ax3.set_xlim(-20, 30)
    ax3.legend(fontsize=8, loc='upper left')
    ax3.grid(True, alpha=0.3)

    # NO concentration on twin axis
    no_ppm = simout['y'][8, :]  # [ppm]
    ax3b = ax3.twinx()
    ax3b.plot(ca, no_ppm, linewidth=1.5, color='tab:green', label='NO')
    ax3b.set_ylabel('NO [ppm]', color='tab:green')
    ax3b.tick_params(axis='y', labelcolor='tab:green')
    ax3b.legend(fontsize=8, loc='upper right')

    fig.suptitle(inj_str, fontsize=10)
    fig.tight_layout()
    plt.savefig('cylinder_pressure.png', dpi=150)
    print("  Plot saved to cylinder_pressure.png")
    plt.show()

    return fig
