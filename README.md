# Mechanical Engineering Toolkit

Python calculators, notebooks and a Streamlit dashboard for everyday mechanical-engineering work: stress analysis, beams, vibration, thermodynamic cycles, fluids, machine elements and simple data science on test data. C++, MATLAB and JavaScript/TypeScript examples are included.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

> **Educational use only.** These tools use simplified models. Always check units, assumptions and design codes, and have results verified by a qualified engineer before using them for real equipment.

## Quick start

```bash
git clone <your-repo-url>
cd mechanicalEngineering
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements-dev.txt

python engineering_toolkit.py        # interactive menu with 45 calculators
python mohr_circle.py                # principal stresses + Mohr's circle plot
python projectile_motion.py --speed 40 --angles 30 45 60
streamlit run web_dashboard.py       # all-in-one dashboard
python -m pytest                     # run the test suite
```

## Repository layout

```
engineering_toolkit.py   45 calculators + interactive CLI (importable functions)
mohr_circle.py           MohrsCircle class: principal stresses, max shear, plot
projectile_motion.py     ideal projectile trajectories, CLI and plotting
web_dashboard.py         Streamlit dashboard (calculators, CSV lab, ML, simulations, PDF)
desktop_gui.py           Tkinter GUI (forecast, beam sizing, vibration)
mechanical_ds/           analytics.py (EDA, anomalies, forecast, ODE, sizing) + storage.py (SQLite history)
notebooks/               Jupyter notebooks (see below)
data/                    tensile-test data (CSV/XLSX) and reference image
examples/                cpp/, matlab/, javascript/, frontend/ (TypeScript)
scripts/                 run_dashboard.sh / run_dashboard.bat
tests/                   pytest suite
```

## Calculators (`engineering_toolkit.py`)

| # | Tool | Function |
|---|------|----------|
| 1 | Linear thermal expansion | `thermal_expansion` |
| 2 | Shaft torque from power and RPM | `shaft_torque` |
| 3 | Spring-mass natural frequency | `spring_mass_frequency` |
| 4 | Factor of safety | `factor_of_safety` |
| 5 | Simply supported beam, point load: reactions, shear, moment | `simply_supported_beam_point_load` |
| 6 | Unit conversion (length, force, pressure, energy, power, temperature) | `convert_units` |
| 7 | Reynolds number | `reynolds_number` |
| 8 | Pump hydraulic and input power | `pump_power` |
| 9 | Plane-wall conduction | `conduction_heat_rate` |
| 10 | Gear pair speed and torque | `gear_train_output` |
| 11 | Heat-engine efficiency | `heat_engine_performance` |
| 12 | Rectangular section properties (A, I, S) | `rectangular_section_properties` |
| 13 | Beam deflection, point load | `beam_deflection_point_load` |
| 14 | Euler column buckling | `euler_buckling_load` |
| 15 | Von Mises equivalent stress | `von_mises_stress` |
| 16 | Solid-shaft torsion and sizing | `torsion_solid_shaft`, `shaft_diameter_for_torque` |
| 17 | Thin-wall pressure vessel | `thin_wall_pressure_vessel` |
| 18 | Carnot efficiency | `carnot_efficiency` |
| 19 | Simply supported beam, uniform load (reactions, V, M, deflection) | `simply_supported_beam_udl` |
| 20 | Cantilever, end point load | `cantilever_beam_end_load` |
| 21 | Cantilever, uniform load | `cantilever_beam_udl` |
| 22 | Multi-load beam: any mix of point and uniform loads on a simply supported, cantilever, fixed-fixed or propped-cantilever beam | `beam_multi_load` |
| 23 | Fixed-fixed beam: point load or full-span uniform load | `fixed_fixed_beam_point_load`, `fixed_fixed_beam_udl` |
| 24 | Fatigue: modified Goodman factor of safety (plus yield check) | `goodman_fatigue` |
| 25 | Heat exchanger, LMTD method: LMTD, duty, area, shell-and-tube correction factor F | `heat_exchanger_lmtd`, `log_mean_temperature_difference`, `shell_and_tube_correction_factor` |
| 26 | Heat exchanger, effectiveness-NTU: rating and sizing for six flow arrangements | `heat_exchanger_ntu`, `exchanger_effectiveness`, `exchanger_ntu` |
| 27 | Stress concentration: Kt (hole in a plate), notch sensitivity q and fatigue notch factor Kf | `stress_concentration_factor`, `notch_sensitivity`, `fatigue_notch_factor` |
| 28 | Fatigue: finite life from the S-N curve (mean stress, Kf, target life) | `sn_fatigue_life`, `sn_curve` |
| 29 | Fatigue: Miner cumulative damage over several load blocks | `miner_cumulative_damage` |
| 30 | Continuous (multi-span) beam: any number of spans, pinned/fixed/free ends, point and uniform loads, per-span I | `continuous_beam` |
| 31 | Pipe flow: Darcy-Weisbach pressure drop (laminar / Colebrook, minor losses, elevation, hydraulic power) | `pipe_pressure_drop` |
| 32 | Rolling-bearing life L10 (ball / roller, reliability factor, required rating for a target life) | `bearing_life_l10`, `bearing_equivalent_load` |
| 33 | Bolted joint: preload, tightening torque, bolt/clamp load, proof, load and separation factors | `bolted_joint`, `metric_tensile_stress_area` |
| 34 | Straight rectangular fin: adiabatic, convective or infinite tip; efficiency and effectiveness | `straight_fin` |
| 35 | Internal-flow convection: Nusselt number and h (laminar, Gnielinski, Dittus-Boelter) | `convection_internal_flow` |
| 36 | Circular section properties, solid or hollow (A, I, J, S, plastic Z, radius of gyration) | `circular_section_properties` |
| 37 | Radiation heat transfer to large surroundings (plus linearised h_rad) | `radiation_heat_rate` |
| 38 | Damped vibration: damping ratio, log decrement, free response, magnification and transmissibility | `damped_vibration` |
| 39 | Spur gear pair geometry: diameters, centre distance, contact ratio, undercut check | `spur_gear_geometry` |
| 40 | Spur gear bending (Lewis) and contact (Hertz) stress with factors of safety | `spur_gear_stress`, `lewis_form_factor` |
| 41 | Spur gear module sizing from allowable bending stress, next standard module | `spur_gear_module_for_bending` |

All functions use SI units, validate their inputs (`ValueError` on bad values) and can be imported:

```python
from engineering_toolkit import beam_deflection_point_load, euler_buckling_load, von_mises_stress

beam = beam_deflection_point_load(span_m=4, load_n=10_000, position_m=2,
                                  youngs_modulus_pa=200e9, second_moment_m4=2000e-8)
print(beam["maximum_deflection_m"])                                # 0.00333 m

print(euler_buckling_load(200e9, 100e-8, 3.0, effective_length_factor=0.7))
print(von_mises_stress(100, 50, 30))                               # MPa in, MPa out
```

New in 2.1:

```python
from engineering_toolkit import (beam_multi_load, fixed_fixed_beam_udl, goodman_fatigue,
                                 heat_exchanger_lmtd, shell_and_tube_correction_factor, heat_exchanger_ntu)

# 6 m beam built in at both ends: two point loads (position m, load N) and a partial uniform load (start m, end m, N/m)
beam = beam_multi_load(6.0, [(1.5, 2000), (4.5, 3000)], [(2.0, 5.0, 500)], "fixed_fixed",
                       youngs_modulus_pa=200e9, second_moment_m4=2000e-8)
print(beam["moment_at_left_nm"], beam["maximum_deflection_m"])     # hogging end moment is negative

print(fixed_fixed_beam_udl(4.0, 5000)["moment_at_left_nm"])        # -wL^2/12 = -6667 N·m

# Goodman: Sut = 600 MPa, corrected Se = 200 MPa, stress cycle +300 / -100 MPa (values in Pa)
print(goodman_fatigue(600e6, 200e6, stress_max_pa=300e6, stress_min_pa=-100e6)["goodman_factor_of_safety"])   # 0.857 < 1

# Required area for 50 kW with U = 500 W/m²K, hot 100 -> 60 °C, cold 20 -> 50 °C, counterflow
print(heat_exchanger_lmtd(100, 60, 20, 50, overall_u_w_m2k=500, heat_duty_w=50e3)["required_area_m2"])
f = shell_and_tube_correction_factor(150, 90, 30, 80)               # one shell pass, 2+ tube passes
# Rating by NTU: C_hot = 2000 W/K, C_cold = 4000 W/K, UA = 3000 W/K
print(heat_exchanger_ntu(2000, 4000, 100, 20, ua_w_k=3000)["heat_rate_w"])
```

New in 2.2:

```python
from engineering_toolkit import stress_concentration_factor, fatigue_notch_factor, sn_fatigue_life

# 100 mm wide plate with a 20 mm central hole: Kt on the net section, then Kf for an 800 MPa steel
kt = stress_concentration_factor("circular_hole_finite_plate", width_m=0.100, hole_diameter_m=0.020)["kt"]   # 2.51
kf = fatigue_notch_factor(kt, notch_radius_m=0.010, ultimate_strength_pa=800e6, loading="axial")["kf"]       # 2.40

# Cycles to failure: Sut = 600 MPa, corrected Se = 200 MPa, nominal cycle +150 / -50 MPa, with the notch above
life = sn_fatigue_life(600e6, 200e6, stress_max_pa=150e6, stress_min_pa=-50e6, notch_factor_kf=kf, target_cycles=1e5)
print(life["life_cycles"])                            # about 6.0e4 cycles
print(life["factor_of_safety_at_target"])             # below 1: it will not reach 1e5 cycles
print(life["allowable_amplitude_at_target_pa"])       # nominal amplitude that would just survive 1e5 cycles
```

New in 2.3:

```python
from engineering_toolkit import (pipe_pressure_drop, bearing_life_l10, bolted_joint, metric_tensile_stress_area,
                                 straight_fin, convection_internal_flow, damped_vibration)

# 100 m of 50 mm commercial-steel pipe carrying 4 L/s of water, fittings K = 3
dp = pipe_pressure_drop(0.05, 100, 998, 1.0e-3, flow_rate_m3_s=0.004, minor_loss_k=3)
print(dp["total_pressure_drop_pa"], dp["friction_factor"])

# 6208-size ball bearing: C = 30 kN, P = 6 kN at 1500 rpm, 95 % reliability
print(bearing_life_l10(30_000, 6_000, speed_rpm=1500, reliability_percent=95)["adjusted_life_hours"])

# M10 class-8.8 bolt (proof 580 MPa), joint constant 0.25, 8 kN external load
at = metric_tensile_stress_area(0.010, 0.0015)
joint = bolted_joint(0.010, at, 580e6, 0.25, external_load_n=8000)
print(joint["tightening_torque_nm"], joint["separation_factor"])

print(straight_fin(200, 50, 0.05, 0.004, 0.1, 100, 25)["heat_rate_w"])            # aluminium fin, adiabatic tip
print(convection_internal_flow(1e5, 0.7, 0.05, 0.026)["heat_transfer_coefficient_w_m2k"])
print(damped_vibration(2, 20_000, 40, forcing_frequency_hz=50)["transmissibility"])
```

New in 2.4 (gears):

```python
from engineering_toolkit import spur_gear_geometry, spur_gear_stress, spur_gear_module_for_bending

print(spur_gear_geometry(0.002, 20, 40)["contact_ratio"])                          # 1.64
s = spur_gear_stress(5000, 1500, 0.003, 20, 40, 0.03, allowable_bending_pa=200e6, allowable_contact_pa=600e6)
print(s["bending_factor_of_safety_pinion"], s["contact_factor_of_safety"])
print(spur_gear_module_for_bending(5000, 1500, 20, 100e6)["recommended_standard_module_m"])   # 0.003
```

New in 2.2 (beams):

```python
from engineering_toolkit import continuous_beam, propped_cantilever_udl

# Propped cantilever (fixed left, pin right), 4 m, 1 kN/m: prop 3wL/8 = 1500 N, wall moment -wL^2/8 = -2000 N·m
r = propped_cantilever_udl(4.0, 1000.0, 200e9, 2000e-8)
print(r["reaction_right_n"], r["moment_at_left_nm"], r["maximum_deflection_m"])

# Two 4 m spans, 1 kN/m over both, plus a 10 kN load at 2 m: reactions at the three supports, support moments, maxima
b = continuous_beam([4, 4], [(2.0, 10000)], [(0, 8, 1000)], youngs_modulus_pa=200e9, second_moment_m4=2000e-8)
print(b["reactions_n"], b["moments_at_supports_nm"], b["maximum_sagging_moment_nm"])
# overhang on the right and different stiffness per span: left_end="pinned", right_end="free", second_moment_m4=[4e-5, 2e-5]
```

Model assumptions: linear-elastic, small deflections, ideal supports, constant EI, downward loads. Sagging moment is positive, so cantilever moment diagrams are negative (hogging). Euler buckling applies only to long, slender, initially straight columns. Thin-wall vessel formulas are flagged when r/t < 10. Multi-load and fixed-fixed beams use Euler-Bernoulli theory with constant EI; point loads are applied downward-positive (use a negative value for an upward load) and a load at x = 0 or x = L goes straight into the support. The Goodman check works on a uniaxial stress cycle at the critical point: the endurance limit you pass in must already be corrected for surface, size, load type, temperature and reliability, and any fatigue notch factor must already be applied to the stresses. For a compressive mean stress the check uses the flat line n = Se / sigma_a. Kt comes from two closed-form or curve-fit cases only (a central circular hole in a finite-width plate, Peterson/Heywood fit valid up to d/w = 0.65 and based on the net section; an elliptical hole in a wide plate, Inglis solution); for shoulders, grooves and other geometries read Kt from a handbook and enter it directly. Notch sensitivity uses the Neuber constant fitted to steels of about 345-1724 MPa (or the Peterson constant for bending/axial), and Kf = 1 + q(Kt - 1) is an estimate, so use test data for other materials. The S-N estimate is a straight line on log-log axes from f*Sut at 10^3 cycles (f = 0.9 by default) to the corrected endurance limit at 10^6 cycles; tensile mean stress is handled with the modified Goodman line, Kf is applied to both mean and alternating stress (conservative), a compressive mean is ignored, and lives below 10^3 cycles are flagged as outside the model. Scatter and reliability are not modelled beyond what you put into Se. Continuous beams use the direct stiffness method with Euler-Bernoulli elements: every junction between spans is a pin support at the same level (no settlement), loads may cross supports, I may differ per span, and the ends can be pinned, fixed or free (overhang); shear is reported just to the right of a point load or support. The propped cantilever is fixed at x = 0 and pinned at x = L. Heat-exchanger methods assume constant specific heats and overall U, no heat loss, and no phase change in the LMTD method (use Cr = 0 in the NTU method for boiling or condensing). The six-arrangement NTU relations are standard textbook correlations; the cross-flow (both unmixed) one is an approximation.

## Dashboard

```bash
python -m pip install -r requirements.txt
streamlit run web_dashboard.py     # or scripts/run_dashboard.sh / scripts/run_dashboard.bat
```

- **Calculators:** every tool above, plus Mohr's circle, projectile motion and beam-depth sizing, with interactive Plotly charts.
- **CSV lab:** statistics, correlations, robust (median/MAD) outlier flags, moving-average baseline forecast, missing-data handling, CSV export.
- **ML training:** Linear Regression, Random Forest or Gradient Boosting with train/test split, MAE / RMSE / R², predictions and model export. Only load exported pickle files you trust.
- **Simulations:** mass-spring-damper response and beam-depth sensitivity.
- **Reports:** PDF report generation and a local SQLite history (`mechanical_history.sqlite3`).

The beam-depth sizing is a stress-only model (no deflection, buckling, fatigue or code factors), and the forecast is a simple baseline, not a validated predictor.

## Notebooks (`notebooks/`)

| Notebook | Topic | Extra packages |
|----------|-------|----------------|
| `SFD-BMD.ipynb` | Shear-force and bending-moment diagrams | none |
| `damped_vibration.ipynb` | Damped vibration | none |
| `numerical_kinematics.ipynb` | Numerical force, velocity, acceleration | rich |
| `projectile_motion.ipynb` | Projectile motion | none |
| `vonmises-failure_plot.ipynb` | Von Mises failure criterion | none |
| `math.ipynb` | Engineering mathematics, numerical methods | none |
| `rankine_cycle.ipynb`, `rankine_reheat_cycle.ipynb` | Rankine cycles | pyromat, rich |
| `refrigeration_cycle.ipynb` | Refrigeration cycle | pyromat, rich |
| `tensile-test.ipynb` | Tensile test, stress-strain, true stress (data in `data/`) | japanize-matplotlib |

Install the extras with `python -m pip install -r requirements-notebooks.txt`. Property-library results (pyromat) depend on its unit configuration, so verify them independently.

## Other languages (`examples/`)

- `cpp/engineering_calculator.cpp`: shaft torque. `g++ -std=c++17 examples/cpp/engineering_calculator.cpp -o torque_calculator`
- `matlab/mass_spring_simulation.m`: mass-spring-damper with `ode45`.
- `javascript/`: browser dashboard demo, open `examples/javascript/index.html`.
- `frontend/`: TypeScript version of the same demo (`npm install`, then `npm run check`).

## Tests

```bash
python -m pytest
```

The suite checks textbook cases (Mohr's circle, projectile range, midspan beam deflection `PL³/48EI`, `5wL⁴/384EI` and cantilever tip deflections, fixed-fixed end moments `PL/8` and `wL²/12` with deflections `PL³/192EI` and `wL⁴/384EI`, superposition and equilibrium of multi-load beams, Goodman hand calculations, propped-cantilever and continuous-beam textbook reactions and moments (including the three-moment equation), Kt closed-form values and Kf definitions, S-N anchor points and life round-trips, LMTD and effectiveness-NTU closed forms, NTU/LMTD cross-checks, Euler load, von Mises limits, Carnot efficiency, torsion round-trips) and invalid-input handling. Tests do not replace validation against a trusted reference. CI runs on Python 3.10-3.13 via GitHub Actions.

## Roadmap

- More Kt geometries (shoulder fillets, grooves, transverse holes) and Miner's-rule cumulative damage
- Support settlement, spring supports and concentrated moments for continuous beams

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md). Released under the MIT License, see [LICENSE](LICENSE).


## Machine-element design additions

The Streamlit dashboard now includes four machine-design calculators:
- **Compression spring design** — spring rate, deflection, Wahl-corrected shear stress, solid length and optional free-length clearance.
- **Belt drive design** — ideal belt speed, driven RPM, Euler friction tension ratio, tight/slack tensions and transmitted power.
- **Power screw design** — raising/lowering torque, collar friction, efficiency and a simplified self-locking check.
- **Shaft design (fatigue)** — solid circular shaft diameter using a simplified modified Goodman fatigue criterion and von Mises yield check, including fatigue factors.

All inputs use SI internally; dashboard labels show practical mm/MPa where applicable. These are preliminary educational calculations: validate assumptions, load cases, materials, buckling/clearance, belt type, thread geometry, stress concentration, applicable standards and design codes with a qualified engineer before real-world use.
