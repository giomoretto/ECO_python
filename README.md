# ECO — Automated Calibration using Model-based Optimization

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![acados](https://img.shields.io/badge/acados-v0.5.3-orange)
![CasADi](https://img.shields.io/badge/CasADi-3.7.2-green)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.conengprac.2024.105848-blue)](https://doi.org/10.1016/j.conengprac.2024.105848)

ECO stands for **E**conomic **C**ombustion **O**ptimization and solves for the optimal injector inputs of a direct-injection compression-ignition engine. Using the software package acados, a continuous-time optimal control problem (OCP) is formulated and solved by direct method (multiple shooting). The resulting nonlinear program (NLP) is solved by sequential quadratic programming (SQP) using the solver HPIPM. The resulting Hessian is regularized using Levenberg-Marquardt regularization.

ECO was implemented on a real engine test bench using a rapid prototyping system and embedded controllers. A demonstration is available here: https://vimeo.com/933704668

## 1. Summary

The OCP uses an economic cost function over one high pressure cycle of a cylinder with states $x$. The independent variable is the crank angle $\varphi$; the decision
variables are the injection inputs $u$, which are constant over the horizon.

$$
\begin{aligned}
\underset{{u,\;x(\cdot)}}{\mathrm{min}} \quad & J(u) && \text{(injected fuel energy)}\\
\text{s.t.}\quad
& \frac{\mathrm{d}x}{\mathrm{d}\varphi} = f\big(x(\varphi),u,\varphi\big),
  \qquad \varphi \in [\varphi_0,\varphi_{\mathrm{EVO}}]
  && \text{(cylinder dynamics)}\\
& x(\varphi_0) = x_0 && \text{(initial state)}\\
& h\big(x(\varphi),u,\varphi\big) \le 0
  && \text{(path constraints: } p_{\mathrm{max}},\ \mathrm{d}p,\ \text{CoC)}\\
& h_e\big(x(\varphi_\mathrm{EVO}),u\big) \le 0
  && \text{(terminal: IMEP, } T_\mathrm{EVO},\ \mathrm{NO}_x,\ \Phi,\ \text{inj. spacing)}\\
& u_{\mathrm{min}} \le u \le u_{\mathrm{max}} && \text{(actuator limits)}
\end{aligned}
$$

Minimizing injected fuel at requested IMEP is equivalent to maximizing indicated
efficiency, so the solution is the most fuel-efficient injection strategy that
still respects the mechanical (peak pressure, pressure rise rate) and emission
(NOx, exhaust temperature, equivalence ratio) limits.

## 2. Mathematical Formulation

The NLP as built in
[eco/formulation/create_acados_ocp.py](eco/formulation/create_acados_ocp.py) and
[eco/formulation/init_acados_ocp.py](eco/formulation/init_acados_ocp.py).

### 2.1 Variables

For $n_\mathrm{inj}$ injections the free variables are the start and the duration
of energizing of each injection:

$$
\begin{aligned}
u &= [\mathrm{SOE}_1, \dots, \mathrm{SOE}_{n_\mathrm{inj}},
      \mathrm{DOE}_1, \dots, \mathrm{DOE}_{n_\mathrm{inj}}]^\top
   \in \mathbb{R}^{2 n_\mathrm{inj}}
   && \text{(injection inputs)}\\
x &= [p_\mathrm{cyl}, Q_\mathrm{comb}, \mathrm{IMEP},
      \Theta_\mathrm{uz}, x_\mathrm{NO}]^\top
   && \text{(cylinder states)}
\end{aligned}
$$
<!-- 
SOE is given in degCA aTDC and DOE in µs; the entries of $x$ are in Pa, J, Pa, K
and dimensionless. The last two states are only present when `en_nox = True`. -->

Since $u$ is constant over the horizon, it is carried as an augmented state with
zero dynamics rather than as a control. The acados state, the ODE it integrates,
and the parameter vector are

$$
\begin{aligned}
\tilde{x} &= [u^\top, x^\top, \varphi]^\top && \text{(augmented state)}\\
\frac{\mathrm{d}\tilde{x}}{\mathrm{d}\varphi}
 &= [0^\top, f_\mathrm{CAM}(x,u,\varphi)^\top, 1]^\top = f(\tilde{x})
 && \text{(augmented dynamics)}\\
p &= [\Delta t_\mathrm{inj}] && \text{(parameter)}
\end{aligned}
$$

so the problem has **no controls**: the only free decision is the injection part
of $\tilde{x}$ at the first node. $f_\mathrm{CAM}$ is the crank-angle-resolved
cylinder model in [eco/model_casadi/](eco/model_casadi/). 
<!-- Every
$\mathrm{max}(a,b)$ and saturation in it is replaced by a smooth surrogate

$$
\mathrm{smax}(a,b) = \frac{1}{2}\left(a + b + \sqrt{\epsilon + (a-b)^2}\right),
$$

so $f_\mathrm{CAM}$ is twice differentiable and the SQP method obtains usable
derivatives everywhere. All variables entering acados are affinely scaled,
$\tilde{v} = (v - v_\mathrm{offs}) / v_\mathrm{scale}$. -->

### 2.2 Constraints and slacks

$$
\begin{aligned}
h(\tilde{x}) &= [p_\mathrm{cyl}, \mathrm{d}p_\mathrm{cyl}, h_\mathrm{CoC}]^\top
   && \text{(path constraints)}\\
h_e(\tilde{x},p) &= [\mathrm{IMEP}, \Theta_\mathrm{EVO},
   \mathrm{NO}_\mathrm{ppm}, \Phi, b_\mathrm{inj}]^\top
   && \text{(terminal constraints)}
\end{aligned}
$$

$\mathrm{d}p_\mathrm{cyl}$ is short for the crank-angle derivative
$\mathrm{d}p_\mathrm{cyl} / \mathrm{d}\varphi$, i.e. the pressure rise rate in
bar/degCA (`dpCyl` in the code).

| Constraint | Bound | Default | Type | Slacked |
|---|---|---|---|---|
| Peak pressure | $0 \le p_\mathrm{cyl} \le p_\mathrm{max}$ | 150 bar | path constraint | — |
| Pressure rise rate | $\mathrm{d}p_\mathrm{cyl} \le \mathrm{d}p_\mathrm{max}$ | 4 bar/degCA | path constraint | yes |
| Center of combustion | $Q_\mathrm{comb} \ge Q_\mathrm{tot}/2$ for $\varphi \ge \mathrm{CoC}_\mathrm{max}$ | 20 degCA aTDC | path constraint | — |
| Indicated mean effective pressure | $\mathrm{IMEP} \ge \mathrm{IMEP}_\mathrm{ref}$ | 6 bar | terminal constraint | — |
| Exhaust gas temperature | $\Theta_\mathrm{EVO} \ge \Theta_\mathrm{min}$ | swept (0…540 °C) | terminal constraint | — |
| NOx concentration | $0 \le \mathrm{NO}_\mathrm{ppm} \le c_{\mathrm{NO}_x}$ | swept (10000…900 ppm) | terminal constraint | yes |
| Equivalence ratio | $0 \le \Phi \le \Phi_\mathrm{max}$ | 1/1.3 | terminal constraint | — |
| Injection spacing | $b_{\mathrm{inj},i} \ge 0$ | $\Delta t_\mathrm{inj} = 400$ µs | terminal constraint | — |

The pressure rise rate and the NOx concentration are the two constraints that can
render the QP infeasible from a poor initial guess. They are therefore relaxed by
non-negative slack variables $s_l^k, s_u^k \ge 0$, which widen the bounds and are
penalized in the cost by

$$
\rho(s) = \sum_{k} \left( Z_l (s_l^k)^2 + z_l s_l^k
                        + Z_u (s_u^k)^2 + z_u s_u^k \right),
$$

with $(Z_l, z_l, Z_u, z_u) = (10^3, 10^3, 20, 1)$ for the pressure rise rate
(`idxsh = [1]`) and $(10^3, 10^3, 10, 10)$ for NOx (`idxsh_e`). The large
linear weights $z_l, z_u$ keep the slacks at zero whenever the original
constraint is attainable, so the relaxation is exact at the solution.

### 2.3 The NLP

The OCP is transcribed by **multiple shooting** with an **explicit fourth-order
Runge-Kutta** integrator on a non-uniform crank-angle grid: fine steps
($\Delta\varphi = 0.5^\circ$) over the optimization range, e.g.
$[-16^\circ, 26^\circ]$, then coarse steps ($3^\circ$) up to EVO at $155^\circ$,
so the terminal constraints are evaluated where they physically apply. With one
RK4 step over interval $k$ written as
$\tilde{x}^{k+1} = F_\mathrm{RK4}(\tilde{x}^k, p; \Delta\varphi_k)$, the
optimization variables are the shooting states and the slacks,

$$
\xi = [\tilde{x}^{0\top}, \dots, \tilde{x}^{N\top},
       s_l^{1}, s_u^{1}, \dots, s_l^{N}, s_u^{N}]^\top ,
$$

and the NLP that acados hands to the SQP method is

$$
\begin{aligned}
\underset{\xi}{\mathrm{min}} \quad
& Q_\mathrm{tot}(u) + \rho(s)
  && \text{(scaled fuel energy + slack penalty)}\\
\text{s.t.}\quad
& \tilde{x}^{k+1} - F_\mathrm{RK4}(\tilde{x}^k,p) = 0,
  \qquad k = 0,\dots,N-1
  && \text{(shooting gaps)}\\
& b_\mathrm{lb} \le \tilde{x}^0 \le b_\mathrm{ub}
  && \text{(initial-state box)}\\
& h_\mathrm{lb} - s_l^k \le h(\tilde{x}^k) \le h_\mathrm{ub} + s_u^k,
  \qquad k = 1,\dots,N-1
  && \text{(relaxed path constraints)}\\
& h_{e,\mathrm{lb}} - s_l^N \le h_e(\tilde{x}^N,p) \le h_{e,\mathrm{ub}} + s_u^N
  && \text{(relaxed terminal constraints)}\\
& s_l^k \ge 0, \quad s_u^k \ge 0
  && \text{(slack non-negativity)}
\end{aligned}
$$

The economic cost $Q_\mathrm{tot}(u) = m_\mathrm{tot}(u) H_u$ is the total
injected fuel energy, evaluated algebraically from the injector model and
therefore a function of $u$ alone. It enters as a **terminal cost only**; the
stage cost is zero.

<!-- The divisor $s_Q$ is not an optimization variable but a fixed normalization
constant — the same scaling factor that is applied to the heat-release state
$Q_\mathrm{comb}$, i.e. the `'QComb'` entry of `par_opt.scale`
($s_Q = 10^3$ J in all example scripts). Dividing by it expresses the objective
in the same scaled units as $Q_\mathrm{comb}$ and keeps it of order one, which
matters because the Levenberg-Marquardt term $\lambda_\mathrm{LM} I$ is added
with a fixed absolute weight. The same constant normalizes the center-of-
combustion constraint, $h_\mathrm{CoC} = Q_\mathrm{comb}/s_Q - Q_\mathrm{tot}/(2 s_Q)$. -->

The initial-state box
$b_\mathrm{lb}, b_\mathrm{ub}$ leaves $u$ free within
$[u_\mathrm{min}, u_\mathrm{max}]$ and pins the physical states to
$x(\varphi_0)$, obtained by forward-simulating from intake valve closing
($-172^\circ$) to the start of the optimization range.

For the default two-injection, NOx-enabled setup this gives
$\tilde{x} \in \mathbb{R}^{10}$ on $N+1 = 128$ shooting nodes, i.e. roughly
$1.3\cdot10^3$ optimization variables.

### 2.4 SQP and Levenberg-Marquardt regularization

The NLP is **not** solved by an interior-point method such as IPOPT. It is solved
by **sequential quadratic programming** (`nlp_solver_type = 'SQP'`; `'SQP_RTI'`
performs a single real-time iteration when `n_sqp_max == 1`). Collecting the
equality constraints in $g(\xi)$ and the inequality constraints in $c(\xi)$, the
Lagrangian is

$$
\mathcal{L}(\xi,\mu,\nu) = J(\xi) + \mu^\top g(\xi) + \nu^\top c(\xi),
$$

and each SQP iteration $i$ solves the QP subproblem

$$
\begin{aligned}
\underset{\Delta\xi}{\mathrm{min}} \quad
& \frac{1}{2} \Delta\xi^\top B_i \Delta\xi + \nabla J(\xi_i)^\top \Delta\xi
  && \text{(quadratic model of the cost)}\\
\text{s.t.}\quad
& \nabla g(\xi_i)^\top \Delta\xi + g(\xi_i) = 0
  && \text{(linearized equalities)}\\
& \nabla c(\xi_i)^\top \Delta\xi + c(\xi_i) \le 0
  && \text{(linearized inequalities)}
\end{aligned}
$$

followed by the full step $\xi_{i+1} = \xi_i + \alpha \Delta\xi_i$ with
$\alpha = 1$. Each QP is reduced by **partial condensing** to a horizon of five
stages and solved by **HPIPM**; HPIPM is itself an interior-point solver, but it
acts on the condensed QP subproblems, not on the nonlinear program.

**Levenberg-Marquardt regularization.** Setting `exact_hess_dyn = 0` and
`exact_hess_constr = 0` drops the indefinite second-order terms of the dynamics
and constraints from $\nabla^2_{\xi\xi}\mathcal{L}$, leaving the exact cost
Hessian, to which a Levenberg-Marquardt term is added:

$$
B_i = \nabla^2_{\xi\xi} J(\xi_i) + \lambda_\mathrm{LM} I,
\qquad \lambda_\mathrm{LM} = 10^{-2}.
$$

This keeps $B_i \succ 0$, so every QP is convex and well-posed for HPIPM, and no
further regularization is applied (`regularize_method = 'NO_REGULARIZE'`).
Iterations stop once the KKT residual falls below the tolerances, or after
`n_sqp_max` iterations.
<!-- 
| Option | Value |
|---|---|
| `nlp_solver_type` | `SQP` (`SQP_RTI` if `n_sqp_max == 1`) |
| `qp_solver` | `PARTIAL_CONDENSING_HPIPM` |
| `qp_solver_cond_N` | `5` (condensed horizon) |
| `nlp_solver_max_iter` | `par_opt.sqp['n_sqp_max']` |
| `nlp_solver_step_length` | `par_opt.sqp['step_size']` (`1.0`) |
| `levenberg_marquardt` | `1e-2` |
| `hessian_approx` | `EXACT`, with `exact_hess_dyn = 0`, `exact_hess_constr = 0` |
| `regularize_method` | `NO_REGULARIZE` |
| `integrator_type` | `ERK`, `sim_method_num_stages = 4` |
| tolerances | stationarity `1e-4`; equality / inequality / complementarity `1e-6` | -->

## 3. Running the Code

### 3.1 Setup

ECO requires a built [acados](https://github.com/acados/acados) installation with
its Python interface — follow the official
[installation guide](https://docs.acados.org/python_interface/index.html).

**Its location is the only path you need to configure**, in `python_code/.env`:

```bash
ACADOS_SOURCE_DIR=/path/to/acados
```

This is the directory containing acados' `lib/` and `include/`. Everything else
— the shared libraries to preload, `LD_LIBRARY_PATH` for the generated C code —
is derived from it by [eco/acados_env.py](eco/acados_env.py), which every entry
point calls via `load_acados()`. Exporting `ACADOS_SOURCE_DIR` as a shell
variable works too and takes precedence; if neither is set, `../acados` and
`~/acados` are tried before an error is raised.

Then create the environment and install ECO with its dependencies, which are
declared in [pyproject.toml](pyproject.toml):

```bash
cd python_code
python3 -m venv env
source env/bin/activate
pip install -e ".[dev]"
```

Finally install the acados Python interface **from the tree you configured
above** — it generates and compiles C code against the acados library, so it has
to come from the same installation rather than from PyPI:

```bash
pip install -e "$ACADOS_SOURCE_DIR/interfaces/acados_template"
```

`load_acados()` checks this on every run: it raises if `acados_template` is
missing, and warns if it was imported from somewhere other than
`ACADOS_SOURCE_DIR`, since a mismatch between the Python interface and the
compiled library produces confusing build errors.

### 3.2 Scripts to run

The scripts can be run from any working directory. Each one generates and
compiles C code into a `c_generated_code*` folder on first execution, which takes
a while; later runs reuse it.

| Order | Script | What it does |
|---|---|---|
| 1 | [examples/simulate_acados_and_plot.py](examples/simulate_acados_and_plot.py) | Forward simulation only (IVC → EVO) with fixed injection inputs. Use this first to check the model and the acados build. Writes `cylinder_pressure.png`. |
| 2 | [examples/main_python.py](examples/main_python.py) | **Main entry point.** Pre-integrates IVC → OCP start, builds the OCP, solves it unconstrained, then runs Pareto sweeps over $\Theta_\mathrm{min}$ and $c_{\mathrm{NO}_x}$. Writes `examples/pareto_results.png`. |
| 3 | [examples/optimize_two_injections.py](examples/optimize_two_injections.py) | Single operating point, two injections, with a homotopy that ramps the constraints from loose to target values and reports the tightest feasible solution. Use this when a direct solve at the target constraints fails. |
| 4 | [examples/example_injection_optimization.py](examples/example_injection_optimization.py) | Minimal, self-contained demonstration of the formulation API. |

```bash
python examples/simulate_acados_and_plot.py
python examples/main_python.py
python examples/optimize_two_injections.py
```

To obtain optimal injector inputs for your own operating point, edit the
`OptimizationParameters` class inside the example script (references, bounds,
$u_0$, optimization range, scaling) and the `OperatingPoint`
([eco/simulation/par_op_def.py](eco/simulation/par_op_def.py): engine speed,
intake pressure/temperature, burnt gas fraction, rail pressure), then rerun.

> Initialize the solver with an initial guess close to a previously obtained
> solution. Otherwise robust convergence to a feasible solution is not guaranteed.
<!-- 
### 3.3 Call sequence

Any custom driver script follows the same three steps:

```python
from eco.formulation.create_acados_ocp import create_acados_functions_inj_opt
from eco.formulation.run_sqp_acados import run_sqp_acados_inj_opt

# 1. build + compile the OCP (once)
ocp_solver, fcn = create_acados_functions_inj_opt(par_opt, par_model, par_op)

# 2. + 3. initialize bounds/guess and solve (repeat per reference value)
u_opt, status, result = run_sqp_acados_inj_opt(
    ocp_solver, fcn, par_model, par_sim, par_opt)
```

`run_sqp_acados_inj_opt` calls `create_init_acados_ocp_inj_opt` internally.
It returns the optimal $u = [\mathrm{SOE}_1,\dots,\mathrm{DOE}_{n_\mathrm{inj}}]$
in physical units, the acados status (`0` = success), and the full unscaled
trajectories in `result['x']`, `result['u']`, `result['ca']`.

The simulation-based initial trajectory is set only on the very first solve.
Subsequent solves keep the previous solution as their guess and only refresh the
constraint bounds, so a tightening reference is approached gradually.

### 3.4 Tests

```bash
python -m pytest tests/test_model_casadi.py -v
``` -->

## 4. Package Layout

```
eco/
├── model_casadi/                 # Physical models, CasADi SX throughout
│   ├── model_parameters.py       # Engine geometry, thermodynamics, calibration
│   ├── complete_model.py         # Full RHS: combustion + optional NOx
│   ├── in_cylinder_model.py      # In-cylinder thermodynamics
│   ├── combustion_model.py       # Heat release
│   ├── algebraic_injector_model.py
│   ├── ign_del_model.py          # Total ignition delay (chem. + phys.)
│   ├── ignition_delay_joerg.py   # Chemical ignition delay correlation
│   ├── utils.py                  # Smooth validity/saturation helpers
│   └── subfunctions/             # cyl. volume, kappa, Woschni, two-zone, Zeldovich
│
├── simulation/                   # Forward simulation
│   ├── par_op_def.py             # Operating point (IVC conditions, valve timings)
│   ├── export_complete_model.py  # AcadosModel export for the integrator
│   ├── acados_simulation.py      # AcadosSimSolver-based integration (RK4 / Euler)
│   └── complete_simulation.py    # Simulation over a crank-angle grid
│
└── formulation/                  # Optimal control problem
    ├── casadi_model.py           # Bridge to the symbolic model
    ├── scale_unscale.py          # Affine variable scaling
    ├── create_acados_ocp.py      # OCP: cost, constraints, solver options
    ├── init_acados_ocp.py        # Bounds, references, warm start
    └── run_sqp_acados.py         # Solve + unscale results
```
