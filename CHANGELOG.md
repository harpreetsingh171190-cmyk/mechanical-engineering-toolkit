# Changelog

## 2.6.1 - Test fix

### Fixed
- `tests/test_spur_gears.py::test_contact_ratio_and_undercut_flag` asserted that a 17-tooth pinion is undercut-free, which contradicts the 20-degree limit of 2/sin^2(20 deg) = 17.1 teeth that the same test checks. The test now uses 18 teeth for the undercut-free case and asserts that 17 teeth are flagged. No calculation code changed.
- The full test suite now passes.

## 2.6.0 - Structural analysis and heat-exchanger tools surfaced in dashboard

### Added / verified
- Dashboard calculator menu includes **Multi-load beam** with point loads and partial/full UDLs, and selectable simply-supported, cantilever, fixed-fixed and propped-cantilever supports; plots shear-force, bending-moment and deflection diagrams.
- **Fixed-fixed beam** calculator supports off-centre point loads and full-span UDLs, with reactions, end moments and deflection plots.
- **Fatigue (modified Goodman)** calculator reports mean/alternating stress, fatigue factor of safety and optional first-cycle yield check, with a Goodman diagram.
- **Heat exchanger (LMTD)** calculator supports counterflow, parallel flow and shell-and-tube correction factor, and can solve LMTD, duty or required area.
- **Heat exchanger (effectiveness-NTU)** calculator supports rating from UA and sizing from target effectiveness, with effectiveness-versus-NTU plot.
- Added release-level verification notes; engineering assumptions and limitations are documented in README.md.

### Validation
- Focused beam, fatigue and heat-exchanger test suites are run for this release.
- The full suite still has a pre-existing failing assertion in `tests/test_spur_gears.py::test_contact_ratio_and_undercut_flag`; this is unrelated to these tools and is recorded rather than hidden.


## 2.4.0 - Spur gear design

### Added
- `spur_gear_geometry`: standard full-depth pair (pitch, outside, root and base diameters, centre distance, circular and base pitch, transverse contact ratio, undercut check).
- `spur_gear_stress`: tangential force, Barth velocity factor (milled, hobbed, ground), Lewis bending stress for pinion and gear (`lewis_form_factor`, 20-degree table with interpolation) and Hertz contact stress, with optional factors of safety. A simplified screening method, not full AGMA/ISO 6336.
- `spur_gear_module_for_bending`: minimum module for an allowable bending stress at a chosen face-width-to-module ratio, plus the next standard module.
- CLI options 39-41, two dashboard calculators (with a stress-versus-module chart) and about 20 new test cases in `tests/test_spur_gears.py`.

## 2.3.0 - Fluids, machine elements, heat transfer and vibration

### Added
- `pipe_pressure_drop`: Darcy-Weisbach pressure drop for a circular pipe from flow rate or velocity (laminar 64/Re, Colebrook-White otherwise, transitional-range flag), with minor losses, elevation change, head loss and hydraulic power.
- `bearing_life_l10` and `bearing_equivalent_load`: basic rating life for ball (p = 3) and roller (p = 10/3) bearings in revolutions and hours, ISO 281 reliability factor a1 (90-99 %), and the minimum dynamic rating needed for a target life.
- `bolted_joint` and `metric_tensile_stress_area`: preload, tightening torque (T = K Fi d), bolt and clamp loads, bolt stress, and proof, load and separation factors from the joint constant.
- `straight_fin`: constant-section rectangular fin with adiabatic, convective or infinite tip; heat rate, tip temperature, efficiency, effectiveness and a thin-fin (Biot) validity flag.
- `convection_internal_flow`: Nusselt number and heat-transfer coefficient in a smooth tube (laminar constants, Gnielinski, Dittus-Boelter) with a validity-range flag.
- `circular_section_properties` (solid or hollow: A, I, J, S, plastic Z, radius of gyration) and `radiation_heat_rate` (grey body to large surroundings, with h_rad).
- `damped_vibration`: natural frequency, damping ratio and type, logarithmic decrement, Q, exact free response for under-, critically and over-damped systems, and magnification and transmissibility under harmonic forcing.
- CLI options 31-38, eight new dashboard calculators (with system curve, load diagram, fin temperature profile, free-response and transmissibility plots) and about 40 new test cases in `tests/test_new_calculators.py`.

## 2.2.0 - Fatigue, stress concentration and more beams

### Added
- `stress_concentration_factor`: theoretical Kt for a central circular hole in a finite-width plate (Peterson/Heywood fit, net-section basis, d/w up to 0.65) and for an elliptical hole in a wide plate (Inglis, Kt = 1 + 2a/b); optionally returns the peak stress.
- `notch_sensitivity` (Neuber or Peterson, steels; bending, axial or torsion) and `fatigue_notch_factor` (Kf = 1 + q(Kt - 1)).
- `sn_fatigue_life`: cycles to failure from a log-log S-N line (f*Sut at 1e3 cycles to Se at 1e6), tensile mean stress via the modified Goodman line, optional Kf, infinite-life and below-1e3-cycle detection, and with a target life the strength, factor of safety and allowable amplitude. `sn_curve` samples the curve for plotting.
- `miner_cumulative_damage`: Palmgren-Miner linear damage over constant-amplitude blocks (CLI option 29).
- `propped_cantilever` support in `beam_multi_load`, plus `propped_cantilever_point_load` and `propped_cantilever_udl`.
- `continuous_beam`: multi-span beams (direct stiffness method) with pinned, fixed or free ends, point and uniform loads across supports, per-span I; returns reactions, support moments, extreme moments and shear, and shear/moment/deflection arrays (CLI option 30, new dashboard calculator).
- CLI options 27-28, two dashboard calculators (Kt/Kf with a Kt-vs-d/w chart, S-N life with a log-log S-N plot), and about 60 new test cases.

### Changed
- `goodman_fatigue` now shares its stress-cycle parsing with the S-N function (`_stress_cycle`); behaviour and error handling are unchanged.

## 2.1.0 - Beams, fatigue and heat exchangers

### Added
- `beam_multi_load`: any number of downward point loads and (partial or full-span) uniform loads on a simply supported, cantilever or fixed-fixed beam, solved exactly by superposition with singularity functions. Returns reactions, end moments, shear, moment, optional deflection, and the true extreme moment and deflection (refined between samples).
- `fixed_fixed_beam_point_load` and `fixed_fixed_beam_udl` (statically indeterminate beams built in at both ends).
- `goodman_fatigue`: modified Goodman factor of safety from max/min or mean/amplitude stress, compressive-mean handling, allowable amplitude and Langer yield check.
- Heat exchangers: `log_mean_temperature_difference`, `heat_exchanger_lmtd` (counterflow or parallel flow, duty / area / U solving), `shell_and_tube_correction_factor`, `exchanger_effectiveness`, `exchanger_ntu` and `heat_exchanger_ntu` (rating and sizing; counterflow, parallel flow, shell-and-tube, three cross-flow cases).
- CLI options 22-26 and five new dashboard calculators (multi-load beam with load tables, fixed-fixed beam, Goodman diagram, LMTD, effectiveness-NTU curve).
- `parse_numeric_rows` helper for load tables; about 120 new test cases (textbook values, superposition, equilibrium, NTU/LMTD cross-checks, invalid input).

## 2.0.0 - Professional cleanup

### Changed
- Reorganised the repository: `notebooks/`, `data/`, `examples/` (C++, MATLAB, JavaScript, TypeScript) and `scripts/`.
- Notebook data paths now point to `../data/` (the tensile-test notebook no longer depends on a path on one person's Desktop).
- `requirements.txt` uses lower bounds instead of exact pins; notebook and dev extras live in separate files.
- Rewrote the README.

### Fixed
- Beam shear diagram: a load applied exactly at a support no longer shows a spurious constant shear force.
- `HistoryStore` closes its SQLite connections (the file used to stay locked, notably on Windows).
- Dashboard: the unit-conversion family selector now updates the From/To lists immediately; the beam-load position field no longer lags behind the span input; the gear-train inputs each have their own column.
- Removed unused imports; corrected the `optimize_beam_depth` docstring (it is a closed-form sizing, not a numerical search).

### Added
- Calculators: rectangular section properties, simply supported beam deflection, Euler column buckling, von Mises stress, solid-shaft torsion and sizing, thin-wall pressure vessels, Carnot efficiency (CLI options 12-18, dashboard, importable functions).
- Beam cases: simply supported beam under a uniform load, cantilever with an end load, cantilever with a uniform load (reactions, shear, moment and optional deflection; CLI options 19-21 and dashboard).
- 27 new tests, GitHub Actions CI (Python 3.10-3.13), `pyproject.toml` with pytest configuration.
