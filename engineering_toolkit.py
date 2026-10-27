"""Extra mechanical-engineering calculators and an interactive command-line menu.

All functions use SI units unless their parameter names state otherwise.
Educational use only: verify assumptions and results before engineering decisions.
"""
from __future__ import annotations

import math


def _finite_positive(name: str, value: float, *, allow_zero: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number.")
    return value


def thermal_expansion(length_m: float, alpha_per_c: float, delta_temp_c: float) -> dict[str, float]:
    """Linear thermal expansion: returns delta_length_m and final_length_m."""
    L = _finite_positive("length_m", length_m, allow_zero=True)
    alpha = _finite_positive("alpha_per_c", alpha_per_c, allow_zero=True)
    dt = float(delta_temp_c)
    if not math.isfinite(dt):
        raise ValueError("delta_temp_c must be finite.")
    change = L * alpha * dt
    return {"delta_length_m": change, "final_length_m": L + change}


def shaft_torque(power_w: float, speed_rpm: float) -> dict[str, float]:
    """Torque from power and rotational speed; returns angular_speed_rad_s and torque_nm."""
    power = _finite_positive("power_w", power_w, allow_zero=True)
    rpm = _finite_positive("speed_rpm", speed_rpm)
    omega = 2.0 * math.pi * rpm / 60.0
    return {"angular_speed_rad_s": omega, "torque_nm": power / omega}


def spring_mass_frequency(stiffness_n_m: float, mass_kg: float) -> dict[str, float]:
    """Undamped single-degree-of-freedom natural frequency."""
    k = _finite_positive("stiffness_n_m", stiffness_n_m)
    m = _finite_positive("mass_kg", mass_kg)
    omega = math.sqrt(k / m)
    return {"angular_frequency_rad_s": omega, "frequency_hz": omega / (2.0 * math.pi),
            "period_s": 2.0 * math.pi / omega}


def factor_of_safety(failure_strength_pa: float, working_stress_pa: float) -> float:
    """Return strength / absolute working stress."""
    strength = _finite_positive("failure_strength_pa", failure_strength_pa)
    stress = _finite_positive("working_stress_pa", abs(float(working_stress_pa)))
    return strength / stress


def simply_supported_beam_point_load(span_m: float, load_n: float, position_m: float,
                                     *, samples: int = 201) -> dict[str, object]:
    """Beam reactions and sampled shear/moment for one downward point load.

    The load is at `position_m` from the left support. Positive sagging moment
    convention is used. Returned arrays are Python lists for easy JSON/export.
    """
    L = _finite_positive("span_m", span_m)
    P = _finite_positive("load_n", load_n, allow_zero=True)
    a = float(position_m)
    if not math.isfinite(a) or not 0 <= a <= L:
        raise ValueError("position_m must lie between 0 and span_m.")
    if not isinstance(samples, int) or samples < 2:
        raise ValueError("samples must be an integer of at least 2.")
    rb = P * a / L
    ra = P - rb
    x = [L * i / (samples - 1) for i in range(samples)]
    # A load applied exactly at a support goes straight into that reaction,
    # so it produces no shear inside the span (a == 0 or a == L).
    shear = [ra - (P if (xi >= a and a < L) else 0.0) for xi in x]
    moment = [ra * xi - (P * (xi - a) if xi >= a else 0.0) for xi in x]
    max_m = P * a * (L - a) / L
    return {"reaction_left_n": ra, "reaction_right_n": rb, "x_m": x,
            "shear_n": shear, "moment_nm": moment, "maximum_moment_nm": max_m}


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert supported length, force, pressure, energy, power, and temperature units."""
    v = float(value)
    if not math.isfinite(v):
        raise ValueError("value must be finite.")
    units = {
        "length": {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "km": 1e3, "in": 0.0254, "ft": 0.3048},
        "force": {"N": 1.0, "kN": 1e3, "lbf": 4.4482216152605},
        "pressure": {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9, "bar": 1e5, "psi": 6894.757293168},
        "energy": {"J": 1.0, "kJ": 1e3, "cal": 4.184, "kWh": 3.6e6},
        "power": {"W": 1.0, "kW": 1e3, "hp": 745.699871582},
    }
    for category, table in units.items():
        if from_unit in table and to_unit in table:
            return v * table[from_unit] / table[to_unit]
    if from_unit in ("C", "F", "K") and to_unit in ("C", "F", "K"):
        c = v if from_unit == "C" else (v - 32) * 5 / 9 if from_unit == "F" else v - 273.15
        return c if to_unit == "C" else c * 9 / 5 + 32 if to_unit == "F" else c + 273.15
    raise ValueError("Unsupported conversion. See README for supported units; units must be in the same category.")



def reynolds_number(density_kg_m3: float, velocity_m_s: float, diameter_m: float,
                   dynamic_viscosity_pa_s: float) -> float:
    """Reynolds number for internal/external flow using characteristic length D."""
    rho = _finite_positive("density_kg_m3", density_kg_m3)
    velocity = _finite_positive("velocity_m_s", abs(float(velocity_m_s)), allow_zero=True)
    diameter = _finite_positive("diameter_m", diameter_m)
    mu = _finite_positive("dynamic_viscosity_pa_s", dynamic_viscosity_pa_s)
    return rho * velocity * diameter / mu


def pump_power(flow_rate_m3_s: float, pressure_rise_pa: float, efficiency: float) -> dict[str, float]:
    """Hydraulic and input power for a pump; efficiency must be in (0, 1]."""
    q = _finite_positive("flow_rate_m3_s", flow_rate_m3_s, allow_zero=True)
    dp = _finite_positive("pressure_rise_pa", pressure_rise_pa, allow_zero=True)
    eta = float(efficiency)
    if not math.isfinite(eta) or not 0 < eta <= 1:
        raise ValueError("efficiency must be greater than 0 and at most 1 (fraction, not percent).")
    hydraulic = q * dp
    return {"hydraulic_power_w": hydraulic, "input_power_w": hydraulic / eta}


def conduction_heat_rate(conductivity_w_mk: float, area_m2: float,
                         temperature_hot_c: float, temperature_cold_c: float,
                         thickness_m: float) -> float:
    """Steady 1-D plane-wall conduction magnitude, Qdot = k A |deltaT| / L."""
    k = _finite_positive("conductivity_w_mk", conductivity_w_mk)
    area = _finite_positive("area_m2", area_m2)
    thickness = _finite_positive("thickness_m", thickness_m)
    th, tc = float(temperature_hot_c), float(temperature_cold_c)
    if not math.isfinite(th) or not math.isfinite(tc):
        raise ValueError("temperatures must be finite.")
    return k * area * abs(th - tc) / thickness


def gear_train_output(input_rpm: float, driver_teeth: int, driven_teeth: int,
                      input_torque_nm: float = 0.0, efficiency: float = 1.0) -> dict[str, float]:
    """Ideal external gear pair: output direction is opposite input; speed/torque magnitudes returned."""
    rpm = _finite_positive("input_rpm", input_rpm, allow_zero=True)
    torque = _finite_positive("input_torque_nm", input_torque_nm, allow_zero=True)
    if not isinstance(driver_teeth, int) or driver_teeth <= 0 or not isinstance(driven_teeth, int) or driven_teeth <= 0:
        raise ValueError("gear tooth counts must be positive integers.")
    eta = float(efficiency)
    if not math.isfinite(eta) or not 0 < eta <= 1:
        raise ValueError("efficiency must be greater than 0 and at most 1.")
    ratio = driven_teeth / driver_teeth
    return {"gear_ratio": ratio, "output_rpm": rpm / ratio,
            "output_torque_nm": torque * ratio * eta}


def heat_engine_performance(heat_input_w: float, heat_rejected_w: float) -> dict[str, float]:
    """Thermal efficiency and net power for a heat engine (Qin > 0, 0 <= Qout <= Qin)."""
    qin = _finite_positive("heat_input_w", heat_input_w)
    qout = _finite_positive("heat_rejected_w", heat_rejected_w, allow_zero=True)
    if qout > qin:
        raise ValueError("heat_rejected_w cannot exceed heat_input_w for this model.")
    net = qin - qout
    return {"net_power_w": net, "thermal_efficiency": net / qin,
            "thermal_efficiency_percent": 100 * net / qin}

def rectangular_section_properties(width_m: float, height_m: float) -> dict[str, float]:
    """Area, second moment of area (bending about the width axis) and section modulus."""
    b = _finite_positive("width_m", width_m)
    h = _finite_positive("height_m", height_m)
    inertia = b * h**3 / 12.0
    return {"area_m2": b * h, "second_moment_m4": inertia, "section_modulus_m3": inertia / (h / 2.0)}


def beam_deflection_point_load(span_m: float, load_n: float, position_m: float,
                               youngs_modulus_pa: float, second_moment_m4: float,
                               *, samples: int = 201) -> dict[str, object]:
    """Deflection of a simply supported beam with one downward point load.

    Classical Euler-Bernoulli result (small deflections, constant EI). Deflection is
    reported as a positive downward magnitude. Returns the sampled curve plus the
    analytical maximum deflection and its location from the left support.
    """
    L = _finite_positive("span_m", span_m)
    P = _finite_positive("load_n", load_n, allow_zero=True)
    a = float(position_m)
    if not math.isfinite(a) or not 0 <= a <= L:
        raise ValueError("position_m must lie between 0 and span_m.")
    E = _finite_positive("youngs_modulus_pa", youngs_modulus_pa)
    I = _finite_positive("second_moment_m4", second_moment_m4)
    if not isinstance(samples, int) or samples < 2:
        raise ValueError("samples must be an integer of at least 2.")
    b = L - a
    ei = E * I

    def deflection(x: float) -> float:
        if x <= a:
            return P * b * x * (L**2 - b**2 - x**2) / (6.0 * ei * L)
        r = L - x
        return P * a * r * (L**2 - a**2 - r**2) / (6.0 * ei * L)

    x = [L * i / (samples - 1) for i in range(samples)]
    short = min(a, b)
    max_def = P * short * (L**2 - short**2) ** 1.5 / (9.0 * math.sqrt(3.0) * ei * L)
    x_max = math.sqrt((L**2 - b**2) / 3.0) if a >= b else L - math.sqrt((L**2 - a**2) / 3.0)
    return {"x_m": x, "deflection_m": [deflection(xi) for xi in x],
            "maximum_deflection_m": max_def, "maximum_deflection_location_m": x_max,
            "midspan_deflection_m": deflection(L / 2.0)}


def euler_buckling_load(youngs_modulus_pa: float, second_moment_m4: float, length_m: float,
                        effective_length_factor: float = 1.0) -> dict[str, float]:
    """Euler critical load P_cr = pi^2 E I / (K L)^2 for an ideal slender column.

    K: 1.0 pinned-pinned, 0.5 fixed-fixed, 0.7 fixed-pinned, 2.0 fixed-free.
    Valid only for long, slender columns that stay elastic.
    """
    E = _finite_positive("youngs_modulus_pa", youngs_modulus_pa)
    I = _finite_positive("second_moment_m4", second_moment_m4)
    L = _finite_positive("length_m", length_m)
    K = _finite_positive("effective_length_factor", effective_length_factor)
    effective = K * L
    return {"critical_load_n": math.pi**2 * E * I / effective**2, "effective_length_m": effective}


def von_mises_stress(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
    """Equivalent (von Mises) stress for plane stress; use consistent stress units."""
    sx, sy, txy = float(sigma_x), float(sigma_y), float(tau_xy)
    if not all(math.isfinite(v) for v in (sx, sy, txy)):
        raise ValueError("stresses must be finite.")
    return math.sqrt(sx**2 - sx * sy + sy**2 + 3.0 * txy**2)


def torsion_solid_shaft(torque_nm: float, diameter_m: float) -> dict[str, float]:
    """Maximum shear stress tau = 16 T / (pi d^3) and polar moment J for a solid round shaft."""
    T = _finite_positive("torque_nm", torque_nm, allow_zero=True)
    d = _finite_positive("diameter_m", diameter_m)
    return {"max_shear_stress_pa": 16.0 * T / (math.pi * d**3),
            "polar_moment_m4": math.pi * d**4 / 32.0}


def shaft_diameter_for_torque(torque_nm: float, allowable_shear_pa: float) -> float:
    """Minimum solid-shaft diameter (m) so that 16 T / (pi d^3) equals the allowable shear stress."""
    T = _finite_positive("torque_nm", torque_nm)
    tau = _finite_positive("allowable_shear_pa", allowable_shear_pa)
    return (16.0 * T / (math.pi * tau)) ** (1.0 / 3.0)


def thin_wall_pressure_vessel(pressure_pa: float, radius_m: float, thickness_m: float) -> dict[str, object]:
    """Membrane stresses in thin-walled vessels (gauge pressure, mean radius).

    Cylinder: hoop = p r / t, longitudinal = p r / (2 t). Sphere: p r / (2 t).
    The thin-wall assumption is normally accepted when r / t >= 10.
    """
    p = _finite_positive("pressure_pa", pressure_pa, allow_zero=True)
    r = _finite_positive("radius_m", radius_m)
    t = _finite_positive("thickness_m", thickness_m)
    return {"cylinder_hoop_stress_pa": p * r / t, "cylinder_longitudinal_stress_pa": p * r / (2.0 * t),
            "sphere_stress_pa": p * r / (2.0 * t), "radius_to_thickness": r / t,
            "thin_wall_valid": r / t >= 10.0}


def carnot_efficiency(hot_temperature_k: float, cold_temperature_k: float) -> float:
    """Maximum possible heat-engine efficiency between two reservoirs (absolute temperatures)."""
    th = _finite_positive("hot_temperature_k", hot_temperature_k)
    tc = _finite_positive("cold_temperature_k", cold_temperature_k)
    if tc >= th:
        raise ValueError("cold_temperature_k must be lower than hot_temperature_k.")
    return 1.0 - tc / th


def _beam_stiffness(youngs_modulus_pa, second_moment_m4):
    """Return E*I, or None when neither is given; both must be supplied together."""
    if youngs_modulus_pa is None and second_moment_m4 is None:
        return None
    if youngs_modulus_pa is None or second_moment_m4 is None:
        raise ValueError("Provide both youngs_modulus_pa and second_moment_m4, or neither.")
    return (_finite_positive("youngs_modulus_pa", youngs_modulus_pa)
            * _finite_positive("second_moment_m4", second_moment_m4))


def _check_samples(samples: int) -> None:
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 2:
        raise ValueError("samples must be an integer of at least 2.")


def simply_supported_beam_udl(span_m: float, load_n_per_m: float,
                              youngs_modulus_pa: float | None = None,
                              second_moment_m4: float | None = None,
                              *, samples: int = 201) -> dict[str, object]:
    """Simply supported beam under a uniformly distributed downward load w (N/m).

    Sagging moment positive. Shear is the value of the left free body: V = w (L/2 - x),
    M = w x (L - x) / 2. If E and I are both given, downward deflection (positive) is
    returned too: y = w x (L^3 - 2 L x^2 + x^3) / (24 E I), maximum 5 w L^4 / (384 E I).
    """
    L = _finite_positive("span_m", span_m)
    w = _finite_positive("load_n_per_m", load_n_per_m, allow_zero=True)
    ei = _beam_stiffness(youngs_modulus_pa, second_moment_m4)
    _check_samples(samples)
    x = [L * i / (samples - 1) for i in range(samples)]
    result: dict[str, object] = {
        "reaction_left_n": w * L / 2.0, "reaction_right_n": w * L / 2.0, "x_m": x,
        "shear_n": [w * (L / 2.0 - xi) for xi in x],
        "moment_nm": [w * xi * (L - xi) / 2.0 for xi in x],
        "maximum_moment_nm": w * L**2 / 8.0,
    }
    if ei is not None:
        result["deflection_m"] = [w * xi * (L**3 - 2 * L * xi**2 + xi**3) / (24.0 * ei) for xi in x]
        result["maximum_deflection_m"] = 5.0 * w * L**4 / (384.0 * ei)
    return result


def cantilever_beam_end_load(length_m: float, load_n: float,
                             youngs_modulus_pa: float | None = None,
                             second_moment_m4: float | None = None,
                             *, samples: int = 201) -> dict[str, object]:
    """Cantilever (fixed at x = 0, free at x = L) with a downward point load P at the free end.

    Reaction force P and fixed-end moment P L (hogging, so the moment diagram is negative):
    M(x) = -P (L - x). With E and I: y = P x^2 (3 L - x) / (6 E I), tip deflection P L^3 / (3 E I).
    """
    L = _finite_positive("length_m", length_m)
    P = _finite_positive("load_n", load_n, allow_zero=True)
    ei = _beam_stiffness(youngs_modulus_pa, second_moment_m4)
    _check_samples(samples)
    x = [L * i / (samples - 1) for i in range(samples)]
    result: dict[str, object] = {
        "reaction_force_n": P, "fixed_end_moment_nm": P * L, "x_m": x,
        "shear_n": [P for _ in x], "moment_nm": [-P * (L - xi) for xi in x],
        "maximum_moment_nm": P * L,
    }
    if ei is not None:
        result["deflection_m"] = [P * xi**2 * (3 * L - xi) / (6.0 * ei) for xi in x]
        result["maximum_deflection_m"] = P * L**3 / (3.0 * ei)
    return result


def cantilever_beam_udl(length_m: float, load_n_per_m: float,
                        youngs_modulus_pa: float | None = None,
                        second_moment_m4: float | None = None,
                        *, samples: int = 201) -> dict[str, object]:
    """Cantilever (fixed at x = 0) under a uniformly distributed downward load w (N/m).

    Reaction w L, fixed-end moment w L^2 / 2 (hogging): V = w (L - x), M = -w (L - x)^2 / 2.
    With E and I: y = w x^2 (6 L^2 - 4 L x + x^2) / (24 E I), tip deflection w L^4 / (8 E I).
    """
    L = _finite_positive("length_m", length_m)
    w = _finite_positive("load_n_per_m", load_n_per_m, allow_zero=True)
    ei = _beam_stiffness(youngs_modulus_pa, second_moment_m4)
    _check_samples(samples)
    x = [L * i / (samples - 1) for i in range(samples)]
    result: dict[str, object] = {
        "reaction_force_n": w * L, "fixed_end_moment_nm": w * L**2 / 2.0, "x_m": x,
        "shear_n": [w * (L - xi) for xi in x],
        "moment_nm": [-w * (L - xi) ** 2 / 2.0 for xi in x],
        "maximum_moment_nm": w * L**2 / 2.0,
    }
    if ei is not None:
        result["deflection_m"] = [w * xi**2 * (6 * L**2 - 4 * L * xi + xi**2) / (24.0 * ei) for xi in x]
        result["maximum_deflection_m"] = w * L**4 / (8.0 * ei)
    return result


# ---------------------------------------------------------------------------
# Multi-load beams (superposition) and fixed-fixed beams
# ---------------------------------------------------------------------------

BEAM_SUPPORTS = ("simply_supported", "cantilever", "fixed_fixed", "propped_cantilever")


def _macaulay(x: float, p: float, n: int) -> float:
    """Macaulay bracket <x - p>^n (zero for x < p; n = 0 is a unit step)."""
    if x < p:
        return 0.0
    return 1.0 if n == 0 else (x - p) ** n


def _clean_point_loads(point_loads, L):
    rows = []
    for item in point_loads or ():
        try:
            pos, mag = item
        except (TypeError, ValueError) as exc:
            raise ValueError("Each point load must be a (position_m, load_n) pair.") from exc
        pos, mag = float(pos), float(mag)
        if not math.isfinite(pos) or not 0 <= pos <= L:
            raise ValueError("Point-load positions must lie between 0 and the span.")
        if not math.isfinite(mag):
            raise ValueError("Point loads must be finite numbers.")
        rows.append((pos, mag))
    return rows


def _clean_distributed_loads(distributed_loads, L):
    rows = []
    for item in distributed_loads or ():
        try:
            start, end, w = item
        except (TypeError, ValueError) as exc:
            raise ValueError("Each distributed load must be a (start_m, end_m, load_n_per_m) triple.") from exc
        start, end, w = float(start), float(end), float(w)
        if not all(math.isfinite(v) for v in (start, end, w)):
            raise ValueError("Distributed-load values must be finite numbers.")
        if not 0 <= start < end <= L:
            raise ValueError("Distributed loads need 0 <= start < end <= span.")
        rows.append((start, end, w))
    return rows


def _golden_extremum(fn, lo: float, hi: float, *, iterations: int = 60) -> float:
    """Location of the largest fn on [lo, hi] (unimodal bracket assumed)."""
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(iterations):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = fn(d)
    return (a + b) / 2.0


def beam_multi_load(span_m: float, point_loads=None, distributed_loads=None,
                    support: str = "simply_supported",
                    youngs_modulus_pa: float | None = None,
                    second_moment_m4: float | None = None,
                    *, samples: int = 201) -> dict[str, object]:
    """Beam under any number of downward point loads and uniform distributed loads.

    The result is the exact superposition of every load (Macaulay singularity functions,
    constant EI, small deflections, Euler-Bernoulli theory).

    point_loads        iterable of (position_m, load_n), measured from the left end
    distributed_loads  iterable of (start_m, end_m, load_n_per_m); a full-span UDL is (0, L, w)
    support            "simply_supported" (pin at x = 0 and x = L),
                       "cantilever" (fixed at x = 0, free at x = L) or
                       "fixed_fixed" (built in at both ends; statically indeterminate) or
                       "propped_cantilever" (fixed at x = 0, simple support at x = L; indeterminate)

    Sign conventions: loads are positive downward, sagging moment is positive (so hogging end
    moments are negative), shear is the left free-body force V = dM/dx, deflection is a positive
    downward magnitude. Deflection is returned only when E and I are both given. Reactions and
    bending moments of a fixed-fixed beam do not depend on EI (constant EI assumed).
    """
    L = _finite_positive("span_m", span_m)
    if support not in BEAM_SUPPORTS:
        raise ValueError(f"support must be one of {', '.join(BEAM_SUPPORTS)}.")
    ei = _beam_stiffness(youngs_modulus_pa, second_moment_m4)
    _check_samples(samples)
    points = _clean_point_loads(point_loads, L)
    spreads = _clean_distributed_loads(distributed_loads, L)
    if not points and not spreads:
        raise ValueError("Provide at least one point load or distributed load.")

    # Load terms of the bending moment, M(x) = M_A + R_A x + sum(c <x - p>^n); downward loads
    # give negative c. A UDL from s to e is a load w starting at s minus a load w starting at e.
    terms = [(-mag, pos, 1) for pos, mag in points]
    for start, end, w in spreads:
        terms += [(-w / 2.0, start, 2), (w / 2.0, end, 2)]
    total_load = sum(m for _, m in points) + sum(w * (e - s) for s, e, w in spreads)
    moment_about_left = sum(m * p for p, m in points) + sum(w * (e * e - s * s) / 2.0 for s, e, w in spreads)

    def bracket_sum(x: float, shift: int) -> float:
        # sum of c <x-p>^(n+shift) / ((n+1)...(n+shift)): the term integrated `shift` times
        total = 0.0
        for c, p, n in terms:
            factor = 1.0
            for k in range(1, shift + 1):
                factor *= n + k
            total += c * _macaulay(x, p, n + shift) / factor
        return total

    if support == "simply_supported":
        m_a = 0.0
        r_a = (total_load * L - moment_about_left) / L
    elif support == "cantilever":
        r_a = total_load
        m_a = -moment_about_left
    elif support == "propped_cantilever":   # fixed at x = 0, pin at x = L: M(L) = 0 and deflection(L) = 0
        s0, s2 = bracket_sum(L, 0), bracket_sum(L, 2)
        r_a = 3.0 * (s2 - s0 * L**2 / 2.0) / L**3
        m_a = -s0 - r_a * L
    else:  # fixed_fixed: slope and deflection vanish at both ends
        s1, s2 = bracket_sum(L, 1), bracket_sum(L, 2)
        det = L * (L**3 / 6.0) - (L**2 / 2.0) * (L**2 / 2.0)
        m_a = (-s1 * (L**3 / 6.0) + s2 * (L**2 / 2.0)) / det
        r_a = (-s2 * L + s1 * (L**2 / 2.0)) / det
    # slope at the left end (EI * d'(0), downward positive) so that the deflection closes at x = L
    slope0 = 0.0 if support != "simply_supported" else (r_a * L**3 / 6.0 + bracket_sum(L, 2)) / L

    def moment(x: float) -> float:
        return m_a + r_a * x + sum(c * _macaulay(x, p, n) for c, p, n in terms)

    def shear(x: float) -> float:
        value = r_a
        for c, p, n in terms:
            if n == 1:
                if x >= p and p < L:      # a load at the right end goes into the reaction
                    value += c
            else:
                value += c * n * _macaulay(x, p, n - 1)
        return value

    def deflection(x: float) -> float:   # E I d(x), downward positive
        return -(m_a * x * x / 2.0 + r_a * x**3 / 6.0 + bracket_sum(x, 2)) + slope0 * x

    critical = {0.0, L} | {p for p, _ in points} | {v for s, e, _ in spreads for v in (s, e)}
    grid = sorted(set([L * i / (samples - 1) for i in range(samples)]) | critical)
    m_vals = [moment(xi) for xi in grid]
    v_vals = [shear(xi) for xi in grid]
    # the extreme moment can sit between samples (zero shear inside a distributed load): refine it
    max_abs_m, x_of_m = max((abs(v), xi) for v, xi in zip(m_vals, grid))
    for i in range(1, len(grid) - 1):
        if (v_vals[i - 1] * v_vals[i + 1] <= 0 and abs(m_vals[i]) >= max(abs(m_vals[i - 1]), abs(m_vals[i + 1])) - 1e-12 * (1 + max_abs_m)):
            sign = 1.0 if m_vals[i] >= 0 else -1.0
            xr = _golden_extremum(lambda t: sign * moment(t), grid[i - 1], grid[i + 1])
            if abs(moment(xr)) > max_abs_m:
                max_abs_m, x_of_m = abs(moment(xr)), xr

    result: dict[str, object] = {
        "support": support, "total_load_n": total_load,
        "reaction_left_n": r_a, "reaction_right_n": total_load - r_a,
        "moment_at_left_nm": m_a, "moment_at_right_nm": moment(L),
        "x_m": grid, "shear_n": v_vals, "moment_nm": m_vals,
        "maximum_moment_nm": max_abs_m, "maximum_moment_location_m": x_of_m,
        "maximum_shear_n": max(abs(v) for v in v_vals),
    }
    if ei is not None:
        d_vals = [deflection(xi) / ei for xi in grid]
        k = max(range(len(grid)), key=lambda i: abs(d_vals[i]))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        xr = _golden_extremum(lambda t: abs(deflection(t)), lo, hi)
        if abs(deflection(xr)) / ei > abs(d_vals[k]):
            max_d, x_d = deflection(xr) / ei, xr
        else:
            max_d, x_d = d_vals[k], grid[k]
        result.update({"deflection_m": d_vals, "maximum_deflection_m": max_d,
                       "maximum_deflection_location_m": x_d})
    return result


def fixed_fixed_beam_point_load(span_m: float, load_n: float, position_m: float,
                                youngs_modulus_pa: float | None = None,
                                second_moment_m4: float | None = None,
                                *, samples: int = 201) -> dict[str, object]:
    """Beam built in at both ends with one downward point load at `position_m` from the left.

    With a = position, b = L - a: fixed-end moments are -P a b^2 / L^2 (left) and
    -P a^2 b / L^2 (right), both hogging; reactions are P b^2 (L + 2a) / L^3 and
    P a^2 (L + 2b) / L^3. Midspan load: deflection P L^3 / (192 E I), end moments P L / 8.
    """
    L = _finite_positive("span_m", span_m)
    P = _finite_positive("load_n", load_n, allow_zero=True)
    a = float(position_m)
    if not math.isfinite(a) or not 0 <= a <= L:
        raise ValueError("position_m must lie between 0 and span_m.")
    return beam_multi_load(L, [(a, P)], None, "fixed_fixed", youngs_modulus_pa, second_moment_m4,
                           samples=samples)


def fixed_fixed_beam_udl(span_m: float, load_n_per_m: float,
                         youngs_modulus_pa: float | None = None,
                         second_moment_m4: float | None = None,
                         *, samples: int = 201) -> dict[str, object]:
    """Beam built in at both ends under a uniform downward load w (N/m) over the whole span.

    Reactions w L / 2, end moments -w L^2 / 12 (hogging), midspan moment +w L^2 / 24,
    maximum deflection w L^4 / (384 E I) at midspan.
    """
    L = _finite_positive("span_m", span_m)
    w = _finite_positive("load_n_per_m", load_n_per_m, allow_zero=True)
    return beam_multi_load(L, None, [(0.0, L, w)], "fixed_fixed", youngs_modulus_pa,
                           second_moment_m4, samples=samples)


def propped_cantilever_point_load(span_m: float, load_n: float, position_m: float,
                                  youngs_modulus_pa: float | None = None,
                                  second_moment_m4: float | None = None,
                                  *, samples: int = 201) -> dict[str, object]:
    """Propped cantilever (built in at x = 0, simple support at x = L) with one downward point load.

    With a = position and b = L - a: prop reaction P a^2 (3L - a) / (2 L^3) and fixed-end moment
    -P a b (L + b) / (2 L^2) (hogging). Midspan load: prop 5P/16, wall 11P/16, wall moment -3PL/16.
    """
    L = _finite_positive("span_m", span_m)
    P = _finite_positive("load_n", load_n, allow_zero=True)
    a = float(position_m)
    if not math.isfinite(a) or not 0 <= a <= L:
        raise ValueError("position_m must lie between 0 and span_m.")
    return beam_multi_load(L, [(a, P)], None, "propped_cantilever", youngs_modulus_pa, second_moment_m4,
                           samples=samples)


def propped_cantilever_udl(span_m: float, load_n_per_m: float,
                           youngs_modulus_pa: float | None = None,
                           second_moment_m4: float | None = None,
                           *, samples: int = 201) -> dict[str, object]:
    """Propped cantilever under a uniform downward load w over the whole span.

    Prop reaction 3wL/8, wall reaction 5wL/8, wall moment -wL^2/8 (hogging), largest sagging moment
    9wL^2/128 at x = 5L/8 from the wall, maximum deflection wL^4/(185 E I) at about 0.58 L from the wall (0.42 L from the prop).
    """
    L = _finite_positive("span_m", span_m)
    w = _finite_positive("load_n_per_m", load_n_per_m, allow_zero=True)
    return beam_multi_load(L, None, [(0.0, L, w)], "propped_cantilever", youngs_modulus_pa,
                           second_moment_m4, samples=samples)


# ---------------------------------------------------------------------------
# Continuous (multi-span) beams: direct stiffness method
# ---------------------------------------------------------------------------

CONTINUOUS_END_CONDITIONS = ("pinned", "fixed", "free")


def _solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting; ValueError when the system is singular."""
    n = len(b)
    m = [list(row) + [b[i]] for i, row in enumerate(a)]
    scale = max((abs(m[i][i]) for i in range(n)), default=1.0) or 1.0
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-11 * scale:
            raise ValueError("The beam is unstable: add a support or change the end conditions.")
        m[col], m[piv] = m[piv], m[col]
        row_c = m[col]
        for r in range(col + 1, n):
            f = m[r][col] / row_c[col]
            if f:
                row_r = m[r]
                for c in range(col, n + 1):
                    row_r[c] -= f * row_c[c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
    return x


def continuous_beam(spans_m, point_loads=None, distributed_loads=None, *,
                    left_end: str = "pinned", right_end: str = "pinned",
                    youngs_modulus_pa: float | None = None, second_moment_m4=None,
                    samples: int = 401) -> dict[str, object]:
    """Beam continuous over any number of spans, solved with the direct stiffness method.

    spans_m            lengths of the spans from left to right; every junction between spans is a simple
                       (pin) support on the same level, so 3 spans have 2 interior supports
    point_loads        (position_m, load_n) rows, position measured from the left end of the whole beam
    distributed_loads  (start_m, end_m, load_n_per_m) rows; they may run across supports
    left_end, right_end   "pinned", "fixed" (built in) or "free" (overhang)
    youngs_modulus_pa, second_moment_m4   E and I; I may be one value or one value per span. Without E the
                       I values only set relative stiffness (forces and moments are unchanged); with both
                       the deflection is returned too.

    Euler-Bernoulli theory, small deflections, no support settlement. Loads are positive downward, sagging
    moment is positive (support moments are negative), shear is V = dM/dx (single-valued, taking the value just
    to the right of a point load or support), deflection is a positive downward magnitude. Results hold
    exactly between nodes: moments and shear follow from the element end forces and deflections from the
    exact beam shape functions plus the uniform-load correction.
    Returns reactions (upward positive) at every support, support moments, extreme moments and shear,
    and sampled x / shear / moment (/ deflection) arrays.
    """
    spans = [_finite_positive("span", v) for v in (spans_m if hasattr(spans_m, "__iter__") else [spans_m])]
    if not spans or len(spans) > 50:
        raise ValueError("Provide between 1 and 50 spans.")
    for name, cond in (("left_end", left_end), ("right_end", right_end)):
        if cond not in CONTINUOUS_END_CONDITIONS:
            raise ValueError(f"{name} must be one of {', '.join(CONTINUOUS_END_CONDITIONS)}.")
    _check_samples(samples)
    n_span = len(spans)
    total_len = sum(spans)
    starts = [0.0]
    for v in spans:
        starts.append(starts[-1] + v)

    if youngs_modulus_pa is not None:
        _finite_positive("youngs_modulus_pa", youngs_modulus_pa)
    if second_moment_m4 is None:
        if youngs_modulus_pa is not None:
            raise ValueError("Provide second_moment_m4 together with youngs_modulus_pa.")
        i_list = [1.0] * n_span
    elif hasattr(second_moment_m4, "__iter__"):
        i_list = [_finite_positive("second_moment_m4", v) for v in second_moment_m4]
        if len(i_list) != n_span:
            raise ValueError("Give one second moment per span, or a single value.")
    else:
        i_list = [_finite_positive("second_moment_m4", second_moment_m4)] * n_span
    with_deflection = youngs_modulus_pa is not None
    e_mod = float(youngs_modulus_pa) if with_deflection else 1.0
    ei_span = [e_mod * i for i in i_list]

    points = _clean_point_loads(point_loads, total_len)
    spreads = _clean_distributed_loads(distributed_loads, total_len)
    if not points and not spreads:
        raise ValueError("Provide at least one point load or distributed load.")

    # nodes: supports, every load position and every distributed-load end
    tol = 1e-9 * total_len
    raw = sorted(set(starts) | {p for p, _ in points} | {v for s0, e0, _ in spreads for v in (s0, e0)})
    nodes = [raw[0]]
    for v in raw[1:]:
        if v - nodes[-1] > tol:
            nodes.append(v)
    if len(nodes) > 400:
        raise ValueError("Too many load positions (limit: 400 nodes).")
    n_nodes, n_el = len(nodes), len(nodes) - 1

    def node_of(x: float) -> int:
        return min(range(n_nodes), key=lambda i: abs(nodes[i] - x))

    p_vec = [0.0] * (2 * n_nodes)
    for pos, mag in points:
        p_vec[2 * node_of(pos)] -= mag
    constrained = set()
    for i, x in enumerate(starts):
        k = node_of(x)
        cond = left_end if i == 0 else right_end if i == n_span else "pinned"
        if cond in ("pinned", "fixed"):
            constrained.add(2 * k)
        if cond == "fixed":
            constrained.add(2 * k + 1)
    support_nodes = [node_of(x) for x in starts]

    elements = []      # (x1, length, EI, w, equivalent load vector)
    k_global = [[0.0] * (2 * n_nodes) for _ in range(2 * n_nodes)]
    for e in range(n_el):
        x1, x2 = nodes[e], nodes[e + 1]
        le, xm = x2 - x1, 0.5 * (x1 + x2)
        span_idx = min(max(sum(1 for v in starts[1:-1] if v <= xm), 0), n_span - 1)
        ei = ei_span[span_idx]
        w = sum(wv for s0, e0, wv in spreads if s0 <= xm <= e0)
        p_eq = [-w * le / 2.0, -w * le**2 / 12.0, -w * le / 2.0, w * le**2 / 12.0]
        c = ei / le**3
        ke = [[12 * c, 6 * le * c, -12 * c, 6 * le * c],
              [6 * le * c, 4 * le**2 * c, -6 * le * c, 2 * le**2 * c],
              [-12 * c, -6 * le * c, 12 * c, -6 * le * c],
              [6 * le * c, 2 * le**2 * c, -6 * le * c, 4 * le**2 * c]]
        dofs = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        for a in range(4):
            p_vec[dofs[a]] += p_eq[a]
            for b in range(4):
                k_global[dofs[a]][dofs[b]] += ke[a][b]
        elements.append((x1, le, ei, w, p_eq, ke, dofs))

    free = [d for d in range(2 * n_nodes) if d not in constrained]
    u = [0.0] * (2 * n_nodes)
    if free:
        sol = _solve_linear([[k_global[i][j] for j in free] for i in free], [p_vec[i] for i in free])
        for d, val in zip(free, sol):
            u[d] = val
    ku = [sum(k_global[i][j] * u[j] for j in range(2 * n_nodes)) for i in range(2 * n_nodes)]
    reactions = {d: ku[d] - p_vec[d] for d in constrained}

    ends = []          # per element: (F1, M1) as internal shear and (negative) moment at the left end
    for x1, le, ei, w, p_eq, ke, dofs in elements:
        f_end = [sum(ke[a][b] * u[dofs[b]] for b in range(4)) - p_eq[a] for a in range(4)]
        ends.append((f_end[0], -f_end[1]))

    def locate(x: float) -> int:
        idx = sum(1 for v in nodes[1:-1] if v <= x + tol)
        return min(idx, n_el - 1)

    def shear_at(x: float) -> float:
        e = locate(x)
        return ends[e][0] - elements[e][3] * (x - nodes[e])

    def moment_at(x: float) -> float:
        e = locate(x)
        s_loc = x - nodes[e]
        return ends[e][1] + ends[e][0] * s_loc - elements[e][3] * s_loc**2 / 2.0

    def deflection_at(x: float) -> float:        # downward positive, metres
        e = locate(x)
        x1, le, ei, w, _, _, dofs = elements[e]
        s_loc = x - x1
        xi = s_loc / le
        v1, t1, v2, t2 = (u[d] for d in dofs)
        up = ((1 - 3 * xi**2 + 2 * xi**3) * v1 + le * (xi - 2 * xi**2 + xi**3) * t1
              + (3 * xi**2 - 2 * xi**3) * v2 + le * (-xi**2 + xi**3) * t2)
        return -up + w * s_loc**2 * (le - s_loc)**2 / (24.0 * ei)

    best = {"sag": (-math.inf, 0.0), "hog": (math.inf, 0.0), "shear": 0.0}
    for e, (x1, le, ei, w, *_rest) in enumerate(elements):
        f1, m_left = ends[e]
        cand = [(0.0, m_left), (le, m_left + f1 * le - w * le**2 / 2.0)]
        if w != 0.0 and 0.0 < f1 / w < le:
            s_star = f1 / w
            cand.append((s_star, m_left + f1 * s_star - w * s_star**2 / 2.0))
        for s_loc, m_val in cand:
            if m_val > best["sag"][0]:
                best["sag"] = (m_val, x1 + s_loc)
            if m_val < best["hog"][0]:
                best["hog"] = (m_val, x1 + s_loc)
        best["shear"] = max(best["shear"], abs(f1), abs(f1 - w * le))
    max_abs, x_abs = max((abs(best["sag"][0]), best["sag"][1]), (abs(best["hog"][0]), best["hog"][1]))

    grid = sorted(set([total_len * i / (samples - 1) for i in range(samples)]) | set(nodes))
    load_total = sum(m for _, m in points) + sum(wv * (e0 - s0) for s0, e0, wv in spreads)
    f_noise = 1e-9 * (sum(abs(m) for _, m in points) + sum(abs(wv) * (e0 - s0) for s0, e0, wv in spreads) + 1e-300)
    m_noise = f_noise * total_len

    def tidy(v: float, noise: float) -> float:
        return 0.0 if abs(v) < noise else v

    result: dict[str, object] = {
        "spans_m": spans, "left_end": left_end, "right_end": right_end,
        "total_load_n": load_total,
        "support_positions_m": [starts[i] for i in range(n_span + 1)],
        "reactions_n": [tidy(reactions.get(2 * k, 0.0), f_noise) for k in support_nodes],
        "moments_at_supports_nm": [tidy(moment_at(x), m_noise) for x in starts],
        "moment_at_left_nm": tidy(moment_at(0.0), m_noise), "moment_at_right_nm": tidy(moment_at(total_len), m_noise),
        "maximum_sagging_moment_nm": max(best["sag"][0], 0.0),
        "maximum_sagging_moment_location_m": best["sag"][1],
        "maximum_hogging_moment_nm": min(best["hog"][0], 0.0),
        "maximum_hogging_moment_location_m": best["hog"][1],
        "maximum_moment_nm": max_abs, "maximum_moment_location_m": x_abs,
        "maximum_shear_n": best["shear"],
        "x_m": grid, "shear_n": [shear_at(x) for x in grid], "moment_nm": [moment_at(x) for x in grid],
    }
    if with_deflection:
        d_vals = [deflection_at(x) for x in grid]
        k = max(range(len(grid)), key=lambda i: abs(d_vals[i]))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        xr = _golden_extremum(lambda t: abs(deflection_at(t)), lo, hi)
        if abs(deflection_at(xr)) > abs(d_vals[k]):
            max_d, x_d = deflection_at(xr), xr
        else:
            max_d, x_d = d_vals[k], grid[k]
        result.update({"deflection_m": d_vals, "maximum_deflection_m": max_d, "maximum_deflection_location_m": x_d})
    return result


def parse_numeric_rows(text: str, columns: int) -> list[tuple[float, ...]]:
    """Parse 'a, b' lines (or ';'-separated rows) into tuples of floats; blank text gives [].

    Used by the CLI and dashboard to read load tables, e.g. "1.5, 2000; 3, 500".
    """
    rows = []
    for raw in str(text or "").replace(";", "\n").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = [p for p in raw.replace(",", " ").split() if p]
        if len(parts) != columns:
            raise ValueError(f"Each row needs exactly {columns} numbers, got: '{raw}'.")
        try:
            values = tuple(float(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"Rows must contain only numbers, got: '{raw}'.") from exc
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Rows must contain finite numbers.")
        rows.append(values)
    return rows


# ---------------------------------------------------------------------------
# Fatigue: modified Goodman
# ---------------------------------------------------------------------------

def _stress_cycle(stress_max_pa, stress_min_pa, stress_mean_pa, stress_amplitude_pa) -> tuple[float, float, float, float]:
    """Resolve a uniaxial stress cycle given as (max, min) or (mean, amplitude) -> (max, min, mean, amplitude)."""
    by_extremes = stress_max_pa is not None or stress_min_pa is not None
    by_components = stress_mean_pa is not None or stress_amplitude_pa is not None
    if by_extremes == by_components:
        raise ValueError("Give either stress_max_pa and stress_min_pa, or stress_mean_pa and stress_amplitude_pa.")
    if by_extremes:
        if stress_max_pa is None or stress_min_pa is None:
            raise ValueError("Provide both stress_max_pa and stress_min_pa.")
        s_max, s_min = float(stress_max_pa), float(stress_min_pa)
        if not (math.isfinite(s_max) and math.isfinite(s_min)):
            raise ValueError("Stresses must be finite.")
        if s_max < s_min:
            raise ValueError("stress_max_pa must not be smaller than stress_min_pa.")
        return s_max, s_min, (s_max + s_min) / 2.0, (s_max - s_min) / 2.0
    if stress_mean_pa is None or stress_amplitude_pa is None:
        raise ValueError("Provide both stress_mean_pa and stress_amplitude_pa.")
    s_mean = float(stress_mean_pa)
    s_alt = _finite_positive("stress_amplitude_pa", stress_amplitude_pa, allow_zero=True)
    if not math.isfinite(s_mean):
        raise ValueError("stress_mean_pa must be finite.")
    return s_mean + s_alt, s_mean - s_alt, s_mean, s_alt


def goodman_fatigue(ultimate_strength_pa: float, endurance_limit_pa: float,
                    *, stress_max_pa: float | None = None, stress_min_pa: float | None = None,
                    stress_mean_pa: float | None = None, stress_amplitude_pa: float | None = None,
                    yield_strength_pa: float | None = None) -> dict[str, object]:
    """Infinite-life check with the modified Goodman line for a uniaxial stress cycle.

    Give the cycle as (stress_max_pa, stress_min_pa) or as (stress_mean_pa, stress_amplitude_pa).
    Endurance limit Se must already include surface, size, load and reliability factors.
    Goodman line: sigma_a / Se + sigma_m / Sut = 1/n, so n = 1 / (sigma_a / Se + sigma_m / Sut).
    For a compressive mean stress (sigma_m < 0) the line is flat and n = Se / sigma_a.
    If the yield strength is given, the Langer first-cycle yielding factor Sy / (sigma_a + |sigma_m|
    for tension) is returned too. A static load (zero amplitude) has an infinite fatigue factor.
    The stress cycle should be the effective stress at the critical point, including any
    fatigue stress-concentration factor.
    """
    sut = _finite_positive("ultimate_strength_pa", ultimate_strength_pa)
    se = _finite_positive("endurance_limit_pa", endurance_limit_pa)
    if se > sut:
        raise ValueError("endurance_limit_pa cannot exceed ultimate_strength_pa.")
    s_max, s_min, s_mean, s_alt = _stress_cycle(stress_max_pa, stress_min_pa, stress_mean_pa, stress_amplitude_pa)

    if s_alt == 0.0:
        n = math.inf
    elif s_mean >= 0.0:
        n = 1.0 / (s_alt / se + s_mean / sut)
    else:
        n = se / s_alt
    # largest amplitude the Goodman line permits at this mean stress
    allowable_alt = se * (1.0 - s_mean / sut) if s_mean >= 0.0 else se
    result: dict[str, object] = {
        "stress_max_pa": s_max, "stress_min_pa": s_min,
        "stress_mean_pa": s_mean, "stress_amplitude_pa": s_alt,
        "stress_ratio": (s_min / s_max) if s_max != 0.0 else None,
        "goodman_factor_of_safety": n,
        "allowable_amplitude_at_mean_pa": max(allowable_alt, 0.0),
        "infinite_life_predicted": bool(n >= 1.0),
    }
    if yield_strength_pa is not None:
        sy = _finite_positive("yield_strength_pa", yield_strength_pa)
        peak = max(abs(s_max), abs(s_min))
        result["yield_factor_of_safety"] = sy / peak if peak > 0 else math.inf
    return result


# ---------------------------------------------------------------------------
# Stress concentration (Kt), notch sensitivity and fatigue notch factor (Kf)
# ---------------------------------------------------------------------------

STRESS_CONCENTRATION_GEOMETRIES = ("circular_hole_finite_plate", "elliptical_hole_infinite_plate")
_PLATE_HOLE_MAX_RATIO = 0.65      # d/w range the Peterson/Heywood curve fit is intended for
_KPSI_PA = 6.894757293168e6
_INCH_M = 0.0254


def stress_concentration_factor(geometry: str, *, width_m: float | None = None,
                                hole_diameter_m: float | None = None,
                                semi_axis_transverse_m: float | None = None,
                                semi_axis_parallel_m: float | None = None,
                                nominal_stress_pa: float | None = None) -> dict[str, object]:
    """Theoretical (elastic) stress-concentration factor Kt for a plate under uniaxial tension.

    "circular_hole_finite_plate": central circular hole of diameter d in a plate of width w
        (needs width_m, hole_diameter_m). Peterson/Heywood curve fit
        Kt = 3.00 - 3.13 (d/w) + 3.66 (d/w)^2 - 1.53 (d/w)^3, valid for 0 < d/w <= 0.65, based on the
        NET section: sigma_nom = F / ((w - d) t). Kt = 3 as d/w -> 0.
    "elliptical_hole_infinite_plate": Inglis solution Kt = 1 + 2a/b for an elliptical hole in a wide
        plate, a = semi-axis perpendicular to the load (semi_axis_transverse_m), b = semi-axis parallel
        to the load (semi_axis_parallel_m); sigma_nom is the remote stress. A circle (a = b) gives 3.
    Kt applies to the elastic peak stress at the notch edge; pass nominal_stress_pa to also get
    peak_stress_pa = Kt * nominal. For fatigue use fatigue_notch_factor to convert Kt into Kf.
    """
    if geometry == "circular_hole_finite_plate":
        if width_m is None or hole_diameter_m is None:
            raise ValueError("circular_hole_finite_plate needs width_m and hole_diameter_m.")
        w = _finite_positive("width_m", width_m)
        d = _finite_positive("hole_diameter_m", hole_diameter_m)
        ratio = d / w
        if ratio > _PLATE_HOLE_MAX_RATIO:
            raise ValueError(f"hole_diameter_m / width_m must not exceed {_PLATE_HOLE_MAX_RATIO} for this curve fit.")
        kt = 3.00 - 3.13 * ratio + 3.66 * ratio ** 2 - 1.53 * ratio ** 3
        info = {"diameter_to_width_ratio": ratio, "nominal_stress_basis": "net section F/((w-d)t)"}
    elif geometry == "elliptical_hole_infinite_plate":
        if semi_axis_transverse_m is None or semi_axis_parallel_m is None:
            raise ValueError("elliptical_hole_infinite_plate needs semi_axis_transverse_m and semi_axis_parallel_m.")
        a = _finite_positive("semi_axis_transverse_m", semi_axis_transverse_m)
        b = _finite_positive("semi_axis_parallel_m", semi_axis_parallel_m)
        kt = 1.0 + 2.0 * a / b
        info = {"axis_ratio_a_over_b": a / b, "nominal_stress_basis": "remote stress"}
    else:
        raise ValueError(f"geometry must be one of {STRESS_CONCENTRATION_GEOMETRIES}.")
    result: dict[str, object] = {"geometry": geometry, "kt": kt, **info}
    if nominal_stress_pa is not None:
        nominal = float(nominal_stress_pa)
        if not math.isfinite(nominal):
            raise ValueError("nominal_stress_pa must be finite.")
        result["nominal_stress_pa"] = nominal
        result["peak_stress_pa"] = kt * nominal
    return result


def notch_sensitivity(notch_radius_m: float, ultimate_strength_pa: float, *,
                      loading: str = "bending", method: str = "neuber") -> dict[str, float | str]:
    """Notch sensitivity q (0 = notch has no fatigue effect, 1 = full theoretical Kt).

    method "neuber":   q = 1 / (1 + sqrt(a) / sqrt(r)) with the Shigley/Budynas cubic fit for steels,
        sqrt(a) [sqrt(in)] = 0.246 - 3.08e-3 Sut + 1.51e-5 Sut^2 - 2.67e-8 Sut^3 (bending or axial) or
        0.190 - 2.51e-3 Sut + 1.35e-5 Sut^2 - 2.67e-8 Sut^3 (torsion), Sut in kpsi, fitted for
        about 50-250 kpsi (345-1724 MPa); outside the range of a positive sqrt(a) it raises.
    method "peterson": q = 1 / (1 + a / r) with a = 0.0254 mm * (2070 MPa / Sut)^1.8 (steels, bending or axial only).
    loading is "bending", "axial" or "torsion". This is an empirical estimate for steels: for other
    materials supply Kf from test data instead.
    """
    r = _finite_positive("notch_radius_m", notch_radius_m)
    sut = _finite_positive("ultimate_strength_pa", ultimate_strength_pa)
    if loading not in ("bending", "axial", "torsion"):
        raise ValueError("loading must be 'bending', 'axial' or 'torsion'.")
    if method == "neuber":
        s = sut / _KPSI_PA
        if not 50.0 <= s <= 250.0:
            raise ValueError("The Neuber fit covers about 345-1724 MPa (50-250 kpsi) ultimate strength.")
        if loading == "torsion":
            sqrt_a_in = 0.190 - 2.51e-3 * s + 1.35e-5 * s ** 2 - 2.67e-8 * s ** 3
        else:
            sqrt_a_in = 0.246 - 3.08e-3 * s + 1.51e-5 * s ** 2 - 2.67e-8 * s ** 3
        if sqrt_a_in <= 0.0:
            raise ValueError("Ultimate strength is outside the range where the Neuber fit is positive.")
        q = 1.0 / (1.0 + sqrt_a_in / math.sqrt(r / _INCH_M))
        return {"q": q, "method": "neuber", "loading": loading, "material_constant_sqrt_a_sqrt_in": sqrt_a_in}
    if method == "peterson":
        if loading == "torsion":
            raise ValueError("The Peterson constant here is for bending or axial loading; use method='neuber' for torsion.")
        a_m = 0.0254e-3 * (2070e6 / sut) ** 1.8
        return {"q": 1.0 / (1.0 + a_m / r), "method": "peterson", "loading": loading, "material_constant_a_m": a_m}
    raise ValueError("method must be 'neuber' or 'peterson'.")


def fatigue_notch_factor(kt: float, notch_radius_m: float, ultimate_strength_pa: float, *,
                         loading: str = "bending", method: str = "neuber") -> dict[str, float | str]:
    """Fatigue stress-concentration factor Kf = 1 + q (Kt - 1) from Kt, notch radius and Sut.

    Kf is between 1 and Kt. Multiply the nominal alternating stress by Kf before a fatigue check
    (or pass notch_factor_kf to sn_fatigue_life). See notch_sensitivity for the models and limits.
    """
    kt = float(kt)
    if not math.isfinite(kt) or kt < 1.0:
        raise ValueError("kt must be a finite number of at least 1.")
    sens = notch_sensitivity(notch_radius_m, ultimate_strength_pa, loading=loading, method=method)
    return {**sens, "kt": kt, "kf": 1.0 + float(sens["q"]) * (kt - 1.0)}


# ---------------------------------------------------------------------------
# Finite-life (high-cycle) S-N fatigue
# ---------------------------------------------------------------------------

def _sn_constants(sut: float, se: float, f: float) -> tuple[float, float]:
    """Coefficients of the line Sf = a * N^b through (1e3, f*Sut) and (1e6, Se)."""
    if not 0.0 < f <= 1.0:
        raise ValueError("fatigue_strength_fraction must be in (0, 1].")
    if f * sut <= se:
        raise ValueError("fatigue_strength_fraction * Sut must exceed the endurance limit.")
    return (f * sut) ** 2 / se, -math.log10(f * sut / se) / 3.0


def sn_fatigue_life(ultimate_strength_pa: float, endurance_limit_pa: float, *,
                    stress_max_pa: float | None = None, stress_min_pa: float | None = None,
                    stress_mean_pa: float | None = None, stress_amplitude_pa: float | None = None,
                    notch_factor_kf: float = 1.0, fatigue_strength_fraction: float = 0.9,
                    target_cycles: float | None = None) -> dict[str, object]:
    """Cycles to failure from a straight-line S-N curve on log-log axes (high-cycle fatigue, 1e3 to 1e6).

    The S-N line runs from f * Sut at 1e3 cycles (default f = 0.9) to the corrected endurance limit Se
    at 1e6 cycles: Sf = a N^b with a = (f Sut)^2 / Se and b = -log10(f Sut / Se) / 3. Se must already
    include surface, size, load and reliability factors. The cycle is given as (stress_max_pa, stress_min_pa)
    or (stress_mean_pa, stress_amplitude_pa) as nominal stresses; notch_factor_kf multiplies both the
    mean and the amplitude (conservative). A tensile mean stress is converted to an equivalent fully
    reversed stress with the modified Goodman line, sigma_rev = sigma_a / (1 - sigma_m / Sut); a compressive
    mean is ignored. life_cycles is infinite when sigma_rev <= Se. If sigma_rev >= f Sut the predicted life
    is below 1e3 cycles, where this model is not valid (high_cycle_range_valid is False).
    With target_cycles (>= 1e3) the result also gives the strength at that life, the factor of safety
    on stress and the largest nominal amplitude allowed at the given mean stress.
    """
    sut = _finite_positive("ultimate_strength_pa", ultimate_strength_pa)
    se = _finite_positive("endurance_limit_pa", endurance_limit_pa)
    if se > sut:
        raise ValueError("endurance_limit_pa cannot exceed ultimate_strength_pa.")
    kf = float(notch_factor_kf)
    if not math.isfinite(kf) or kf < 1.0:
        raise ValueError("notch_factor_kf must be a finite number of at least 1.")
    coeff_a, exp_b = _sn_constants(sut, se, float(fatigue_strength_fraction))
    s_max, s_min, s_mean, s_alt = _stress_cycle(stress_max_pa, stress_min_pa, stress_mean_pa, stress_amplitude_pa)
    eff_mean, eff_alt = kf * s_mean, kf * s_alt
    if eff_mean >= sut:
        raise ValueError("Effective mean stress is at or above the ultimate strength (static failure).")
    sigma_rev = eff_alt / (1.0 - eff_mean / sut) if eff_mean > 0.0 else eff_alt

    if sigma_rev <= se:
        life = math.inf
    else:
        life = (sigma_rev / coeff_a) ** (1.0 / exp_b)
    result: dict[str, object] = {
        "stress_max_pa": s_max, "stress_min_pa": s_min, "stress_mean_pa": s_mean, "stress_amplitude_pa": s_alt,
        "notch_factor_kf": kf, "effective_mean_stress_pa": eff_mean, "effective_amplitude_pa": eff_alt,
        "equivalent_reversed_stress_pa": sigma_rev,
        "sn_coefficient_a_pa": coeff_a, "sn_exponent_b": exp_b,
        "fatigue_strength_fraction": float(fatigue_strength_fraction),
        "life_cycles": life, "infinite_life_predicted": math.isinf(life),
        "high_cycle_range_valid": bool(math.isinf(life) or life >= 1e3),
    }
    if target_cycles is not None:
        n = float(target_cycles)
        if not math.isfinite(n) or n < 1e3:
            raise ValueError("target_cycles must be a finite number of at least 1e3.")
        strength = coeff_a * n ** exp_b if n < 1e6 else se
        result["target_cycles"] = n
        result["fatigue_strength_at_target_pa"] = strength
        result["factor_of_safety_at_target"] = strength / sigma_rev if sigma_rev > 0.0 else math.inf
        allowed = strength * (1.0 - eff_mean / sut) if eff_mean > 0.0 else strength
        result["allowable_amplitude_at_target_pa"] = allowed / kf
    return result


def miner_cumulative_damage(ultimate_strength_pa: float, endurance_limit_pa: float, blocks, *,
                            notch_factor_kf: float = 1.0,
                            fatigue_strength_fraction: float = 0.9) -> dict[str, object]:
    """Palmgren-Miner linear damage D = sum(n_i / N_i) for a load history made of constant-amplitude blocks.

    blocks is a list of (stress_mean_pa, stress_amplitude_pa, cycles) rows of nominal stress; each
    block's life N_i comes from sn_fatigue_life (same Se, f and Kf assumptions). Blocks at or below the
    endurance limit add no damage. Failure is predicted when D reaches 1, so blocks_to_failure = 1 / D
    repeats of the whole history. Miner's rule ignores load sequence and is only a rough guide
    (real failures often occur at D between about 0.3 and 3).
    """
    rows = list(blocks)
    if not rows:
        raise ValueError("Provide at least one load block.")
    details, total, valid = [], 0.0, True
    for row in rows:
        if len(row) != 3:
            raise ValueError("Each block must be (stress_mean_pa, stress_amplitude_pa, cycles).")
        mean, amp, n = row
        n = _finite_positive("cycles", n)
        r = sn_fatigue_life(ultimate_strength_pa, endurance_limit_pa, stress_mean_pa=mean, stress_amplitude_pa=amp,
                            notch_factor_kf=notch_factor_kf, fatigue_strength_fraction=fatigue_strength_fraction)
        life = r["life_cycles"]
        damage = 0.0 if math.isinf(life) else n / life
        valid = valid and bool(r["high_cycle_range_valid"])
        total += damage
        details.append({"cycles": n, "life_cycles": life, "damage": damage})
    return {"blocks": details, "total_damage": total,
            "blocks_to_failure": (1.0 / total) if total > 0.0 else math.inf,
            "failure_predicted": bool(total >= 1.0), "high_cycle_range_valid": valid}


def sn_curve(ultimate_strength_pa: float, endurance_limit_pa: float, *,
             fatigue_strength_fraction: float = 0.9, points: int = 60,
             max_cycles: float = 1e7) -> dict[str, object]:
    """Sample the S-N line for plotting: log-spaced cycles from 1e3 to max_cycles and the strength at each.

    Strength follows Sf = a N^b up to 1e6 cycles and stays at the endurance limit beyond.
    """
    sut = _finite_positive("ultimate_strength_pa", ultimate_strength_pa)
    se = _finite_positive("endurance_limit_pa", endurance_limit_pa)
    if se > sut:
        raise ValueError("endurance_limit_pa cannot exceed ultimate_strength_pa.")
    coeff_a, exp_b = _sn_constants(sut, se, float(fatigue_strength_fraction))
    points = int(points)
    if points < 2:
        raise ValueError("points must be at least 2.")
    top = _finite_positive("max_cycles", max_cycles)
    if top <= 1e3:
        raise ValueError("max_cycles must exceed 1e3.")
    step = (math.log10(top) - 3.0) / (points - 1)
    cycles = [10.0 ** (3.0 + i * step) for i in range(points)]
    strength = [coeff_a * n ** exp_b if n < 1e6 else se for n in cycles]
    return {"cycles": cycles, "strength_pa": strength, "sn_coefficient_a_pa": coeff_a, "sn_exponent_b": exp_b}


# ---------------------------------------------------------------------------
# Heat exchangers: LMTD and effectiveness-NTU
# ---------------------------------------------------------------------------

EXCHANGER_ARRANGEMENTS = ("counterflow", "parallel_flow")
NTU_ARRANGEMENTS = ("counterflow", "parallel_flow", "shell_and_tube", "crossflow_both_unmixed",
                    "crossflow_cmax_mixed", "crossflow_cmin_mixed")


def _check_arrangement(arrangement: str, allowed) -> None:
    if arrangement not in allowed:
        raise ValueError(f"arrangement must be one of {', '.join(allowed)}.")


def log_mean_temperature_difference(delta_t1_k: float, delta_t2_k: float) -> float:
    """LMTD = (dT1 - dT2) / ln(dT1 / dT2) for two positive end-point temperature differences."""
    d1 = _finite_positive("delta_t1_k", delta_t1_k)
    d2 = _finite_positive("delta_t2_k", delta_t2_k)
    if abs(d1 - d2) <= 1e-9 * max(d1, d2):
        return (d1 + d2) / 2.0
    return (d1 - d2) / math.log(d1 / d2)


def shell_and_tube_correction_factor(hot_inlet_c: float, hot_outlet_c: float,
                                     cold_inlet_c: float, cold_outlet_c: float) -> float:
    """LMTD correction factor F for one shell pass and 2, 4, ... tube passes.

    P = (Tc,out - Tc,in) / (Th,in - Tc,in), R = (Th,in - Th,out) / (Tc,out - Tc,in).
    Use q = U A F LMTD_counterflow. F <= 1; a low F (below about 0.75) signals a poor
    arrangement, and an infeasible temperature set (a temperature cross) raises ValueError.
    """
    th1, th2, tc1, tc2 = (float(v) for v in (hot_inlet_c, hot_outlet_c, cold_inlet_c, cold_outlet_c))
    if not all(math.isfinite(v) for v in (th1, th2, tc1, tc2)):
        raise ValueError("temperatures must be finite.")
    if th1 <= tc1 or th1 <= th2 or tc2 <= tc1:
        raise ValueError("Need hot inlet > cold inlet, hot inlet > hot outlet and cold outlet > cold inlet.")
    p = (tc2 - tc1) / (th1 - tc1)
    r = (th1 - th2) / (tc2 - tc1)
    s = math.sqrt(r * r + 1.0)
    try:
        if abs(r - 1.0) < 1e-9:
            f = (math.sqrt(2.0) * p / (1.0 - p)) / math.log((2.0 / p - 2.0 + math.sqrt(2.0)) / (2.0 / p - 2.0 - math.sqrt(2.0)))
        else:
            f = (s / (r - 1.0)) * math.log((1.0 - p) / (1.0 - r * p)) / math.log(
                (2.0 / p - 1.0 - r + s) / (2.0 / p - 1.0 - r - s))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("This temperature set is not achievable with one shell pass (F is undefined).") from exc
    return f


def heat_exchanger_lmtd(hot_inlet_c: float, hot_outlet_c: float, cold_inlet_c: float,
                        cold_outlet_c: float, *, arrangement: str = "counterflow",
                        correction_factor: float = 1.0, overall_u_w_m2k: float | None = None,
                        area_m2: float | None = None, heat_duty_w: float | None = None) -> dict[str, float]:
    """Log-mean temperature difference method: q = U A F LMTD.

    arrangement: "counterflow" (dT1 = Th,in - Tc,out, dT2 = Th,out - Tc,in) or
    "parallel_flow" (dT1 = Th,in - Tc,in, dT2 = Th,out - Tc,out).
    correction_factor F (default 1) multiplies the counterflow LMTD for multi-pass or crossflow
    exchangers; see shell_and_tube_correction_factor. Temperatures are in the same unit (degC or K).
    Give two of overall_u_w_m2k, area_m2 and heat_duty_w to solve for the third.
    """
    _check_arrangement(arrangement, EXCHANGER_ARRANGEMENTS)
    th1, th2, tc1, tc2 = (float(v) for v in (hot_inlet_c, hot_outlet_c, cold_inlet_c, cold_outlet_c))
    if not all(math.isfinite(v) for v in (th1, th2, tc1, tc2)):
        raise ValueError("temperatures must be finite.")
    if th2 > th1:
        raise ValueError("The hot stream must cool down (hot outlet <= hot inlet).")
    if tc2 < tc1:
        raise ValueError("The cold stream must heat up (cold outlet >= cold inlet).")
    f = _finite_positive("correction_factor", correction_factor)
    if f > 1.0:
        raise ValueError("correction_factor cannot exceed 1.")
    d1, d2 = (th1 - tc2, th2 - tc1) if arrangement == "counterflow" else (th1 - tc1, th2 - tc2)
    if d1 <= 0 or d2 <= 0:
        raise ValueError("Temperature cross or pinch: both end-point temperature differences must be positive.")
    lmtd = log_mean_temperature_difference(d1, d2)
    given = [v is not None for v in (overall_u_w_m2k, area_m2, heat_duty_w)]
    if all(given):
        raise ValueError("Give at most two of overall_u_w_m2k, area_m2 and heat_duty_w.")
    result: dict[str, float] = {"delta_t1_k": d1, "delta_t2_k": d2, "lmtd_k": lmtd,
                                "correction_factor": f, "effective_lmtd_k": f * lmtd}
    if overall_u_w_m2k is not None and area_m2 is not None:
        u = _finite_positive("overall_u_w_m2k", overall_u_w_m2k)
        a = _finite_positive("area_m2", area_m2)
        result["heat_duty_w"] = u * a * f * lmtd
    elif overall_u_w_m2k is not None and heat_duty_w is not None:
        u = _finite_positive("overall_u_w_m2k", overall_u_w_m2k)
        q = _finite_positive("heat_duty_w", heat_duty_w)
        result["required_area_m2"] = q / (u * f * lmtd)
    elif area_m2 is not None and heat_duty_w is not None:
        a = _finite_positive("area_m2", area_m2)
        q = _finite_positive("heat_duty_w", heat_duty_w)
        result["required_overall_u_w_m2k"] = q / (a * f * lmtd)
    return result


def exchanger_effectiveness(ntu: float, capacity_ratio: float,
                            arrangement: str = "counterflow") -> float:
    """Effectiveness of a heat exchanger from NTU = U A / Cmin and Cr = Cmin / Cmax (0 <= Cr <= 1).

    arrangement: counterflow, parallel_flow, shell_and_tube (one shell pass, 2, 4, ... tube passes),
    crossflow_both_unmixed (common correlation, approximate), crossflow_cmax_mixed
    (Cmax mixed, Cmin unmixed) or crossflow_cmin_mixed (Cmin mixed, Cmax unmixed).
    With Cr = 0 (boiling or condensing) every arrangement gives 1 - exp(-NTU).
    """
    _check_arrangement(arrangement, NTU_ARRANGEMENTS)
    n = _finite_positive("ntu", ntu, allow_zero=True)
    cr = float(capacity_ratio)
    if not math.isfinite(cr) or not 0.0 <= cr <= 1.0:
        raise ValueError("capacity_ratio must lie between 0 and 1 (Cmin / Cmax).")
    if n == 0.0:
        return 0.0
    if cr == 0.0:
        return 1.0 - math.exp(-n)
    if arrangement == "parallel_flow":
        return (1.0 - math.exp(-n * (1.0 + cr))) / (1.0 + cr)
    if arrangement == "counterflow":
        if abs(cr - 1.0) < 1e-12:
            return n / (1.0 + n)
        e = math.exp(-n * (1.0 - cr))
        return (1.0 - e) / (1.0 - cr * e)
    if arrangement == "shell_and_tube":
        s = math.sqrt(1.0 + cr * cr)
        e = math.exp(-n * s)
        return 2.0 / (1.0 + cr + s * (1.0 + e) / (1.0 - e))
    if arrangement == "crossflow_both_unmixed":
        return 1.0 - math.exp((n**0.22 / cr) * (math.exp(-cr * n**0.78) - 1.0))
    if arrangement == "crossflow_cmax_mixed":
        return (1.0 - math.exp(-cr * (1.0 - math.exp(-n)))) / cr
    return 1.0 - math.exp(-(1.0 - math.exp(-cr * n)) / cr)   # crossflow_cmin_mixed


def exchanger_ntu(effectiveness: float, capacity_ratio: float,
                  arrangement: str = "counterflow") -> float:
    """Number of transfer units needed for a target effectiveness (inverse of exchanger_effectiveness).

    Raises ValueError when the effectiveness is not reachable (e.g. parallel flow cannot exceed
    1 / (1 + Cr)). Solved by bisection on the monotonic effectiveness curve.
    """
    _check_arrangement(arrangement, NTU_ARRANGEMENTS)
    eps = float(effectiveness)
    cr = float(capacity_ratio)
    if not math.isfinite(eps) or not 0.0 <= eps < 1.0:
        raise ValueError("effectiveness must be at least 0 and less than 1.")
    if not math.isfinite(cr) or not 0.0 <= cr <= 1.0:
        raise ValueError("capacity_ratio must lie between 0 and 1 (Cmin / Cmax).")
    if eps == 0.0:
        return 0.0
    hi = 1.0
    while exchanger_effectiveness(hi, cr, arrangement) < eps:
        hi *= 2.0
        if hi > 1e6:
            raise ValueError("That effectiveness is not reachable for this arrangement and capacity ratio.")
    lo = 0.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if exchanger_effectiveness(mid, cr, arrangement) < eps:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def heat_exchanger_ntu(hot_capacity_w_k: float, cold_capacity_w_k: float, hot_inlet_c: float,
                       cold_inlet_c: float, *, ua_w_k: float | None = None,
                       effectiveness: float | None = None,
                       arrangement: str = "counterflow") -> dict[str, float]:
    """Effectiveness-NTU analysis. Capacity rates are C = m_dot * cp in W/K.

    Rating problem: give ua_w_k (U A) to get effectiveness, duty and outlet temperatures.
    Sizing problem: give effectiveness to get the required NTU and U A. Exactly one of the two.
    """
    ch = _finite_positive("hot_capacity_w_k", hot_capacity_w_k)
    cc = _finite_positive("cold_capacity_w_k", cold_capacity_w_k)
    th_in, tc_in = float(hot_inlet_c), float(cold_inlet_c)
    if not (math.isfinite(th_in) and math.isfinite(tc_in)):
        raise ValueError("temperatures must be finite.")
    if th_in <= tc_in:
        raise ValueError("hot_inlet_c must be higher than cold_inlet_c.")
    if (ua_w_k is None) == (effectiveness is None):
        raise ValueError("Give exactly one of ua_w_k or effectiveness.")
    c_min, c_max = min(ch, cc), max(ch, cc)
    cr = c_min / c_max
    if ua_w_k is not None:
        ua = _finite_positive("ua_w_k", ua_w_k)
        ntu = ua / c_min
        eps = exchanger_effectiveness(ntu, cr, arrangement)
    else:
        eps = float(effectiveness)
        ntu = exchanger_ntu(eps, cr, arrangement)
        ua = ntu * c_min
    q_max = c_min * (th_in - tc_in)
    q = eps * q_max
    return {"c_min_w_k": c_min, "c_max_w_k": c_max, "capacity_ratio": cr, "ntu": ntu, "ua_w_k": ua,
            "effectiveness": eps, "max_heat_rate_w": q_max, "heat_rate_w": q,
            "hot_outlet_c": th_in - q / ch, "cold_outlet_c": tc_in + q / cc}


# ---- Fluids, machine elements, heat transfer and vibration (added in 2.3.0) -----------------------

GRAVITY_M_S2 = 9.80665
STEFAN_BOLTZMANN_W_M2K4 = 5.670374419e-8


def _colebrook_friction_factor(reynolds: float, relative_roughness: float) -> float:
    """Darcy friction factor from the Colebrook-White equation (fixed-point iteration on 1/sqrt(f))."""
    x = 7.0  # 1/sqrt(f); converges quickly from here for any Re >= 2300 and e/D < 0.05
    for _ in range(100):
        new_x = -2.0 * math.log10(relative_roughness / 3.7 + 2.51 * x / reynolds)
        if abs(new_x - x) < 1e-13 * max(1.0, abs(x)):
            x = new_x
            break
        x = new_x
    return 1.0 / (x * x)


def pipe_pressure_drop(diameter_m: float, length_m: float, density_kg_m3: float,
                       dynamic_viscosity_pa_s: float, *, flow_rate_m3_s: float | None = None,
                       velocity_m_s: float | None = None, roughness_m: float = 4.5e-5,
                       minor_loss_k: float = 0.0, elevation_gain_m: float = 0.0) -> dict[str, object]:
    """Darcy-Weisbach pressure drop in a full circular pipe.

    Give exactly one of ``flow_rate_m3_s`` or ``velocity_m_s``. The friction factor is 64/Re for laminar
    flow (Re < 2300) and the Colebrook-White equation otherwise (the 2300-4000 band is flagged as
    transitional, where any friction factor is only an estimate). ``minor_loss_k`` is the summed loss
    coefficient of fittings and ``elevation_gain_m`` the rise from inlet to outlet. The default roughness
    is commercial steel (0.045 mm). Returns velocity, Reynolds number, friction factor, head losses and
    the total pressure drop (inlet minus outlet, positive means the pump must supply it).
    """
    d = _finite_positive("diameter_m", diameter_m)
    length = _finite_positive("length_m", length_m, allow_zero=True)
    rho = _finite_positive("density_kg_m3", density_kg_m3)
    mu = _finite_positive("dynamic_viscosity_pa_s", dynamic_viscosity_pa_s)
    eps = _finite_positive("roughness_m", roughness_m, allow_zero=True)
    k_minor = _finite_positive("minor_loss_k", minor_loss_k, allow_zero=True)
    dz = float(elevation_gain_m)
    if not math.isfinite(dz):
        raise ValueError("elevation_gain_m must be finite.")
    if (flow_rate_m3_s is None) == (velocity_m_s is None):
        raise ValueError("Give exactly one of flow_rate_m3_s or velocity_m_s.")
    area = math.pi * d * d / 4
    if flow_rate_m3_s is not None:
        q = _finite_positive("flow_rate_m3_s", flow_rate_m3_s, allow_zero=True)
        v = q / area
    else:
        v = _finite_positive("velocity_m_s", velocity_m_s, allow_zero=True)
        q = v * area
    if eps / d >= 0.05:
        raise ValueError("Relative roughness roughness_m/diameter_m must be below 0.05.")
    re = rho * v * d / mu
    if v == 0:
        f, regime = 0.0, "no flow"
    elif re < 2300:
        f, regime = 64.0 / re, "laminar"
    else:
        f = _colebrook_friction_factor(re, eps / d)
        regime = "transitional (friction factor is approximate)" if re < 4000 else "turbulent"
    dynamic = rho * v * v / 2
    friction_dp = f * length / d * dynamic
    minor_dp = k_minor * dynamic
    elevation_dp = rho * GRAVITY_M_S2 * dz
    total = friction_dp + minor_dp + elevation_dp
    return {"flow_rate_m3_s": q, "velocity_m_s": v, "reynolds_number": re, "flow_regime": regime,
            "friction_factor": f, "relative_roughness": eps / d,
            "friction_head_loss_m": friction_dp / (rho * GRAVITY_M_S2),
            "minor_head_loss_m": minor_dp / (rho * GRAVITY_M_S2),
            "friction_pressure_drop_pa": friction_dp, "minor_pressure_drop_pa": minor_dp,
            "elevation_pressure_change_pa": elevation_dp, "total_pressure_drop_pa": total,
            "total_head_m": total / (rho * GRAVITY_M_S2),
            "hydraulic_power_w": total * q}


_BEARING_RELIABILITY = ((90.0, 1.0), (95.0, 0.64), (96.0, 0.55), (97.0, 0.47), (98.0, 0.37), (99.0, 0.25))


def bearing_equivalent_load(radial_load_n: float, axial_load_n: float = 0.0, *,
                            radial_factor_x: float = 1.0, axial_factor_y: float = 0.0) -> float:
    """Equivalent dynamic bearing load P = X*Fr + Y*Fa (X and Y come from the bearing catalogue)."""
    fr = _finite_positive("radial_load_n", radial_load_n, allow_zero=True)
    fa = _finite_positive("axial_load_n", axial_load_n, allow_zero=True)
    x = _finite_positive("radial_factor_x", radial_factor_x, allow_zero=True)
    y = _finite_positive("axial_factor_y", axial_factor_y, allow_zero=True)
    p = x * fr + y * fa
    if p <= 0:
        raise ValueError("The equivalent load must be positive.")
    return p


def bearing_life_l10(dynamic_load_rating_n: float, equivalent_load_n: float, *,
                     speed_rpm: float | None = None, bearing_type: str = "ball",
                     reliability_percent: float = 90.0,
                     target_life_hours: float | None = None) -> dict[str, float]:
    """Rolling-bearing basic rating life L10 = (C/P)^p million revolutions (p = 3 ball, 10/3 roller).

    ``reliability_percent`` (90 to 99) applies the ISO 281 life-modification factor a1, interpolated
    linearly between the tabulated reliabilities 90, 95, 96, 97, 98 and 99 %. With ``speed_rpm`` the life
    is also given in hours; with ``target_life_hours`` (needs speed) the minimum dynamic rating C needed
    for that life at that reliability is returned. Lubrication and contamination factors (a_ISO) are not
    included.
    """
    c = _finite_positive("dynamic_load_rating_n", dynamic_load_rating_n)
    p_load = _finite_positive("equivalent_load_n", equivalent_load_n)
    if bearing_type not in ("ball", "roller"):
        raise ValueError("bearing_type must be 'ball' or 'roller'.")
    exponent = 3.0 if bearing_type == "ball" else 10.0 / 3.0
    rel = float(reliability_percent)
    if not math.isfinite(rel) or not 90.0 <= rel <= 99.0:
        raise ValueError("reliability_percent must be between 90 and 99.")
    a1 = _BEARING_RELIABILITY[-1][1]
    for (r0, a0), (r1, a_1) in zip(_BEARING_RELIABILITY, _BEARING_RELIABILITY[1:]):
        if r0 <= rel <= r1:
            a1 = a0 + (a_1 - a0) * (rel - r0) / (r1 - r0)
            break
    l10 = (c / p_load) ** exponent
    result: dict[str, float] = {"life_exponent": exponent, "l10_million_revolutions": l10,
                                "reliability_factor_a1": a1,
                                "adjusted_life_million_revolutions": a1 * l10}
    if speed_rpm is not None:
        n = _finite_positive("speed_rpm", speed_rpm)
        result["l10_hours"] = l10 * 1e6 / (60 * n)
        result["adjusted_life_hours"] = a1 * result["l10_hours"]
    if target_life_hours is not None:
        if speed_rpm is None:
            raise ValueError("speed_rpm is required together with target_life_hours.")
        hours = _finite_positive("target_life_hours", target_life_hours)
        needed_l10 = hours * 60 * float(speed_rpm) / 1e6 / a1
        result["required_dynamic_rating_n"] = p_load * needed_l10 ** (1 / exponent)
        result["rating_margin"] = c / result["required_dynamic_rating_n"]
    return result


def metric_tensile_stress_area(diameter_m: float, pitch_m: float) -> float:
    """Tensile-stress area of an ISO metric thread: At = pi/4 * (d - 0.9382 p)^2 (m^2)."""
    d = _finite_positive("diameter_m", diameter_m)
    p = _finite_positive("pitch_m", pitch_m)
    if p >= d / 2:
        raise ValueError("pitch_m is unrealistically large for this diameter.")
    return math.pi / 4 * (d - 0.9382 * p) ** 2


def bolted_joint(nominal_diameter_m: float, tensile_stress_area_m2: float, proof_strength_pa: float,
                 joint_constant: float, *, external_load_n: float = 0.0, preload_fraction: float = 0.75,
                 preload_n: float | None = None, torque_coefficient_k: float = 0.2) -> dict[str, float]:
    """Preloaded bolted joint under a tensile external load (Shigley-style joint-diagram analysis).

    ``joint_constant`` C = kb/(kb+km) is the share of the external load carried by the bolt (0 to 1).
    Preload defaults to ``preload_fraction`` of the proof load (0.75 for reused, 0.9 for permanent joints)
    unless ``preload_n`` is given. Tightening torque is T = K*Fi*d. Returns bolt and clamp loads, the
    bolt stress, and three factors: proof factor (bolt stays below proof strength), load factor
    (yield margin on the external load) and separation factor (joint stays clamped). A factor of
    infinity means there is no external load.
    """
    d = _finite_positive("nominal_diameter_m", nominal_diameter_m)
    at = _finite_positive("tensile_stress_area_m2", tensile_stress_area_m2)
    sp = _finite_positive("proof_strength_pa", proof_strength_pa)
    c = float(joint_constant)
    if not math.isfinite(c) or not 0 < c < 1:
        raise ValueError("joint_constant must be strictly between 0 and 1.")
    p = _finite_positive("external_load_n", external_load_n, allow_zero=True)
    k = _finite_positive("torque_coefficient_k", torque_coefficient_k)
    proof_load = sp * at
    if preload_n is None:
        frac = float(preload_fraction)
        if not math.isfinite(frac) or not 0 < frac <= 1:
            raise ValueError("preload_fraction must be in (0, 1].")
        fi = frac * proof_load
    else:
        fi = _finite_positive("preload_n", preload_n)
        if fi > proof_load:
            raise ValueError("preload_n cannot exceed the proof load (proof strength x stress area).")
    bolt_load = fi + c * p
    clamp_load = fi - (1 - c) * p
    inf = math.inf
    return {"proof_load_n": proof_load, "preload_n": fi, "preload_fraction": fi / proof_load,
            "tightening_torque_nm": k * fi * d, "bolt_load_n": bolt_load, "clamp_load_n": clamp_load,
            "bolt_stress_pa": bolt_load / at,
            "proof_factor_of_safety": proof_load / bolt_load,
            "load_factor": (proof_load - fi) / (c * p) if p > 0 else inf,
            "separation_factor": fi / ((1 - c) * p) if p > 0 else inf,
            "joint_stays_clamped": clamp_load > 0}


def straight_fin(conductivity_w_mk: float, heat_transfer_coefficient_w_m2k: float, length_m: float,
                 thickness_m: float, width_m: float, base_temperature_c: float,
                 ambient_temperature_c: float, *, tip: str = "adiabatic") -> dict[str, object]:
    """Rectangular straight fin of constant cross-section (1-D conduction, uniform h).

    ``tip`` is 'adiabatic', 'convective' or 'infinite'. Returns fin heat rate, tip temperature,
    efficiency (against the ideal all-at-base-temperature fin) and effectiveness (against the bare base).
    ``thin_fin_valid`` is False when the Biot number h*t/(2k) exceeds 0.1.
    """
    k = _finite_positive("conductivity_w_mk", conductivity_w_mk)
    h = _finite_positive("heat_transfer_coefficient_w_m2k", heat_transfer_coefficient_w_m2k)
    length = _finite_positive("length_m", length_m)
    t = _finite_positive("thickness_m", thickness_m)
    w = _finite_positive("width_m", width_m)
    tb, ta = float(base_temperature_c), float(ambient_temperature_c)
    if not (math.isfinite(tb) and math.isfinite(ta)):
        raise ValueError("Temperatures must be finite.")
    theta_b = tb - ta
    if theta_b == 0:
        raise ValueError("base_temperature_c and ambient_temperature_c must differ.")
    if tip not in ("adiabatic", "convective", "infinite"):
        raise ValueError("tip must be 'adiabatic', 'convective' or 'infinite'.")
    perimeter, area_c = 2 * (w + t), w * t
    m = math.sqrt(h * perimeter / (k * area_c))
    big_m = math.sqrt(h * perimeter * k * area_c) * theta_b
    ml = m * length
    if tip == "infinite":
        q, tip_ratio, area_s = big_m, 0.0, perimeter * length
    elif tip == "adiabatic":
        q, tip_ratio, area_s = big_m * math.tanh(ml), 1 / math.cosh(ml), perimeter * length
    else:
        beta = h / (m * k)
        q = big_m * (math.sinh(ml) + beta * math.cosh(ml)) / (math.cosh(ml) + beta * math.sinh(ml))
        tip_ratio = 1 / (math.cosh(ml) + beta * math.sinh(ml))
        area_s = perimeter * length + area_c
    return {"fin_parameter_m_per_m": m, "ml": ml, "heat_rate_w": q,
            "tip_temperature_c": ta + theta_b * tip_ratio,
            "efficiency": q / (h * area_s * theta_b), "effectiveness": q / (h * area_c * theta_b),
            "surface_area_m2": area_s, "biot_number": h * t / (2 * k),
            "thin_fin_valid": h * t / (2 * k) <= 0.1}


def convection_internal_flow(reynolds: float, prandtl: float, diameter_m: float,
                             conductivity_w_mk: float, *, correlation: str = "gnielinski",
                             heating: bool = True,
                             wall_condition: str = "constant_temperature") -> dict[str, object]:
    """Nusselt number and heat-transfer coefficient for fully developed flow in a smooth circular tube.

    Laminar (Re < 2300): Nu = 3.66 (constant wall temperature) or 4.36 (constant heat flux).
    Otherwise 'gnielinski' (smooth tube, valid for 3000 <= Re <= 5e6, 0.5 <= Pr <= 2000) or
    'dittus_boelter' (Nu = 0.023 Re^0.8 Pr^n, n = 0.4 heating the fluid and 0.3 cooling it; valid for
    Re >= 10000, 0.6 <= Pr <= 160). ``valid_range`` is False outside the stated limits.
    """
    re = _finite_positive("reynolds", reynolds)
    pr = _finite_positive("prandtl", prandtl)
    d = _finite_positive("diameter_m", diameter_m)
    k = _finite_positive("conductivity_w_mk", conductivity_w_mk)
    if correlation not in ("gnielinski", "dittus_boelter"):
        raise ValueError("correlation must be 'gnielinski' or 'dittus_boelter'.")
    if wall_condition not in ("constant_temperature", "constant_heat_flux"):
        raise ValueError("wall_condition must be 'constant_temperature' or 'constant_heat_flux'.")
    if re < 2300:
        nu, regime, valid, used = (3.66 if wall_condition == "constant_temperature" else 4.36), "laminar", True, "fully developed laminar"
    elif correlation == "dittus_boelter":
        nu = 0.023 * re ** 0.8 * pr ** (0.4 if heating else 0.3)
        regime = "turbulent" if re >= 4000 else "transitional"
        valid, used = re >= 1e4 and 0.6 <= pr <= 160, "Dittus-Boelter"
    else:
        f = (0.790 * math.log(re) - 1.64) ** -2
        nu = (f / 8) * (re - 1000) * pr / (1 + 12.7 * math.sqrt(f / 8) * (pr ** (2 / 3) - 1))
        regime = "turbulent" if re >= 4000 else "transitional"
        valid, used = 3000 <= re <= 5e6 and 0.5 <= pr <= 2000, "Gnielinski"
    return {"nusselt_number": nu, "heat_transfer_coefficient_w_m2k": nu * k / d,
            "flow_regime": regime, "correlation": used, "valid_range": valid}


def circular_section_properties(outer_diameter_m: float, inner_diameter_m: float = 0.0) -> dict[str, float]:
    """Solid (inner diameter 0) or hollow circular section: A, I, J, S, plastic modulus Z and radius of gyration."""
    do = _finite_positive("outer_diameter_m", outer_diameter_m)
    di = _finite_positive("inner_diameter_m", inner_diameter_m, allow_zero=True)
    if di >= do:
        raise ValueError("inner_diameter_m must be smaller than outer_diameter_m.")
    area = math.pi / 4 * (do ** 2 - di ** 2)
    i = math.pi / 64 * (do ** 4 - di ** 4)
    return {"area_m2": area, "second_moment_m4": i, "polar_moment_m4": 2 * i,
            "section_modulus_m3": i / (do / 2), "polar_section_modulus_m3": 2 * i / (do / 2),
            "plastic_modulus_m3": (do ** 3 - di ** 3) / 6, "radius_of_gyration_m": math.sqrt(i / area)}


def radiation_heat_rate(emissivity: float, area_m2: float, surface_temperature_k: float,
                        surroundings_temperature_k: float) -> dict[str, float]:
    """Net grey-body radiation to large surroundings: q = eps*sigma*A*(Ts^4 - Tsur^4), plus the linearised h_rad."""
    eps = float(emissivity)
    if not math.isfinite(eps) or not 0 <= eps <= 1:
        raise ValueError("emissivity must be between 0 and 1.")
    area = _finite_positive("area_m2", area_m2)
    ts = _finite_positive("surface_temperature_k", surface_temperature_k)
    tsur = _finite_positive("surroundings_temperature_k", surroundings_temperature_k)
    h_rad = eps * STEFAN_BOLTZMANN_W_M2K4 * (ts + tsur) * (ts ** 2 + tsur ** 2)
    return {"heat_rate_w": h_rad * area * (ts - tsur), "radiation_coefficient_w_m2k": h_rad}


def damped_vibration(mass_kg: float, stiffness_n_m: float, damping_n_s_m: float = 0.0, *,
                     forcing_frequency_hz: float | None = None, initial_displacement_m: float = 0.0,
                     initial_velocity_m_s: float = 0.0, response_duration_s: float | None = None,
                     samples: int = 400) -> dict[str, object]:
    """Single-degree-of-freedom mass-spring-damper: free decay and harmonic forcing.

    Returns natural frequency, damping ratio, damped frequency, logarithmic decrement and the damping type.
    With ``forcing_frequency_hz`` it adds the dynamic magnification factor and the force/motion
    transmissibility (infinity at resonance with no damping; isolation needs a frequency ratio above
    sqrt(2)). With ``response_duration_s`` it adds the exact free response x(t) from the initial conditions.
    """
    m = _finite_positive("mass_kg", mass_kg)
    k = _finite_positive("stiffness_n_m", stiffness_n_m)
    c = _finite_positive("damping_n_s_m", damping_n_s_m, allow_zero=True)
    x0, v0 = float(initial_displacement_m), float(initial_velocity_m_s)
    if not (math.isfinite(x0) and math.isfinite(v0)):
        raise ValueError("Initial conditions must be finite.")
    wn = math.sqrt(k / m)
    c_crit = 2 * math.sqrt(k * m)
    zeta = c / c_crit
    if zeta == 0:
        kind = "undamped"
    elif zeta < 1 - 1e-9:
        kind = "underdamped"
    elif zeta <= 1 + 1e-9:
        kind = "critically damped"
    else:
        kind = "overdamped"
    underdamped = zeta < 1 - 1e-9
    wd = wn * math.sqrt(1 - zeta ** 2) if underdamped else 0.0
    result: dict[str, object] = {
        "natural_frequency_hz": wn / (2 * math.pi), "natural_frequency_rad_s": wn,
        "critical_damping_n_s_m": c_crit, "damping_ratio": zeta, "damping_type": kind,
        "damped_frequency_hz": wd / (2 * math.pi),
        "logarithmic_decrement": 2 * math.pi * zeta / math.sqrt(1 - zeta ** 2) if underdamped else None,
        "quality_factor": 1 / (2 * zeta) if zeta > 0 else math.inf}
    if forcing_frequency_hz is not None:
        f = _finite_positive("forcing_frequency_hz", forcing_frequency_hz, allow_zero=True)
        r = 2 * math.pi * f / wn
        denom = math.sqrt((1 - r * r) ** 2 + (2 * zeta * r) ** 2)
        if denom < 1e-12:
            mag = trans = math.inf
        else:
            mag = 1 / denom
            trans = math.sqrt(1 + (2 * zeta * r) ** 2) * mag
        result.update({"frequency_ratio": r, "magnification_factor": mag, "transmissibility": trans,
                       "isolates_vibration": bool(trans < 1)})
    if response_duration_s is not None:
        duration = _finite_positive("response_duration_s", response_duration_s)
        _check_samples(samples)
        ts = [duration * i / (samples - 1) for i in range(samples)]
        s = zeta * wn
        if underdamped:
            b = (v0 + s * x0) / wd
            xs = [math.exp(-s * t) * (x0 * math.cos(wd * t) + b * math.sin(wd * t)) for t in ts]
        elif kind == "critically damped":
            xs = [(x0 + (v0 + wn * x0) * t) * math.exp(-wn * t) for t in ts]
        else:
            root = wn * math.sqrt(zeta ** 2 - 1)
            s1, s2 = -s + root, -s - root
            a = (v0 - s2 * x0) / (s1 - s2)
            xs = [a * math.exp(s1 * t) + (x0 - a) * math.exp(s2 * t) for t in ts]
        result.update({"time_s": ts, "displacement_m": xs})
    return result


# ---- Spur gear design (added in 2.4.0) ---------------------------------------------------------------

_LEWIS_Y_20DEG = ((12, 0.245), (13, 0.261), (14, 0.277), (15, 0.290), (16, 0.296), (17, 0.303), (18, 0.309),
                  (19, 0.314), (20, 0.322), (21, 0.328), (22, 0.331), (24, 0.337), (26, 0.346), (28, 0.353),
                  (30, 0.359), (34, 0.371), (38, 0.384), (43, 0.397), (50, 0.409), (60, 0.422), (75, 0.435),
                  (100, 0.447), (150, 0.460), (300, 0.472))
STANDARD_MODULES_MM = (0.5, 0.6, 0.8, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50)


def _teeth(name: str, value) -> int:
    n = int(value)
    if n != value or n < 3:
        raise ValueError(f"{name} must be a whole number of teeth (at least 3).")
    return n


def lewis_form_factor(teeth: int) -> float:
    """Lewis form factor Y for 20-degree full-depth teeth (Shigley table, linear interpolation; 12 teeth minimum)."""
    n = _teeth("teeth", teeth)
    if n < 12:
        raise ValueError("The Lewis table used here starts at 12 teeth (fewer teeth are normally undercut).")
    if n >= 300:
        return _LEWIS_Y_20DEG[-1][1]
    for (n0, y0), (n1, y1) in zip(_LEWIS_Y_20DEG, _LEWIS_Y_20DEG[1:]):
        if n0 <= n <= n1:
            return y0 + (y1 - y0) * (n - n0) / (n1 - n0)
    raise ValueError("teeth out of range.")  # pragma: no cover


def _velocity_factor(pitch_velocity_m_s: float, finish: str) -> float:
    v = pitch_velocity_m_s
    if finish == "milled":
        return (3.05 + v) / 3.05
    if finish == "hobbed":
        return (6.1 + v) / 6.1
    if finish == "ground":
        return (3.56 + math.sqrt(v)) / 3.56
    raise ValueError("finish must be 'milled', 'hobbed' or 'ground'.")


def spur_gear_geometry(module_m: float, pinion_teeth: int, gear_teeth: int, *,
                       pressure_angle_deg: float = 20.0) -> dict[str, object]:
    """Standard full-depth external spur-gear pair (addendum m, dedendum 1.25 m, no profile shift).

    Returns pitch, outside, root and base diameters, centre distance, circular and base pitch, ratio, the
    transverse contact ratio and a check against undercutting (minimum teeth 2/sin^2(phi)).
    """
    m = _finite_positive("module_m", module_m)
    n1, n2 = _teeth("pinion_teeth", pinion_teeth), _teeth("gear_teeth", gear_teeth)
    phi_deg = float(pressure_angle_deg)
    if not math.isfinite(phi_deg) or not 10 <= phi_deg <= 30:
        raise ValueError("pressure_angle_deg must be between 10 and 30.")
    phi = math.radians(phi_deg)
    d1, d2 = m * n1, m * n2
    rb1, rb2 = d1 / 2 * math.cos(phi), d2 / 2 * math.cos(phi)
    ra1, ra2 = d1 / 2 + m, d2 / 2 + m
    centre = (d1 + d2) / 2
    contact = (math.sqrt(ra1 ** 2 - rb1 ** 2) + math.sqrt(ra2 ** 2 - rb2 ** 2) - centre * math.sin(phi)) / (math.pi * m * math.cos(phi))
    min_teeth = 2 / math.sin(phi) ** 2
    return {"pinion_pitch_diameter_m": d1, "gear_pitch_diameter_m": d2, "centre_distance_m": centre,
            "pinion_outside_diameter_m": d1 + 2 * m, "gear_outside_diameter_m": d2 + 2 * m,
            "pinion_root_diameter_m": d1 - 2.5 * m, "gear_root_diameter_m": d2 - 2.5 * m,
            "pinion_base_diameter_m": 2 * rb1, "gear_base_diameter_m": 2 * rb2,
            "circular_pitch_m": math.pi * m, "base_pitch_m": math.pi * m * math.cos(phi),
            "whole_depth_m": 2.25 * m, "speed_ratio": n2 / n1, "contact_ratio": contact,
            "minimum_teeth_no_undercut": min_teeth, "undercut_free": n1 >= min_teeth}


def spur_gear_stress(power_w: float, pinion_rpm: float, module_m: float, pinion_teeth: int, gear_teeth: int,
                     face_width_m: float, *, finish: str = "hobbed", application_factor: float = 1.0,
                     pinion_modulus_pa: float = 200e9, gear_modulus_pa: float = 200e9,
                     poisson_ratio: float = 0.3, allowable_bending_pa: float | None = None,
                     allowable_contact_pa: float | None = None) -> dict[str, object]:
    """Lewis bending stress (Barth velocity factor) and Hertz contact stress for a 20-degree spur pair.

    sigma_b = Wt*Ko*Kv/(F*m*Y) at the pinion and gear, sigma_c = Cp*sqrt(Wt*Ko*Kv/(F*d1*I)) with
    I = sin(phi)cos(phi)/2 * mG/(mG+1). This is a simplified screening method; it omits the AGMA/ISO
    geometry, dynamic, load-distribution and life factors. Optional allowable stresses add factors of safety.
    """
    p = _finite_positive("power_w", power_w)
    n = _finite_positive("pinion_rpm", pinion_rpm)
    m = _finite_positive("module_m", module_m)
    n1, n2 = _teeth("pinion_teeth", pinion_teeth), _teeth("gear_teeth", gear_teeth)
    f = _finite_positive("face_width_m", face_width_m)
    ko = _finite_positive("application_factor", application_factor)
    e1, e2 = _finite_positive("pinion_modulus_pa", pinion_modulus_pa), _finite_positive("gear_modulus_pa", gear_modulus_pa)
    nu = float(poisson_ratio)
    if not math.isfinite(nu) or not 0 <= nu < 0.5:
        raise ValueError("poisson_ratio must be in [0, 0.5).")
    d1 = m * n1
    omega = 2 * math.pi * n / 60
    torque = p / omega
    wt = 2 * torque / d1
    v = omega * d1 / 2
    kv = _velocity_factor(v, finish)
    y1, y2 = lewis_form_factor(n1), lewis_form_factor(n2)
    load = wt * ko * kv
    phi = math.radians(20)
    mg = n2 / n1
    i_geo = math.sin(phi) * math.cos(phi) / 2 * mg / (mg + 1)
    cp = math.sqrt(1 / (math.pi * ((1 - nu ** 2) / e1 + (1 - nu ** 2) / e2)))
    sb1, sb2 = load / (f * m * y1), load / (f * m * y2)
    sc = cp * math.sqrt(load / (f * d1 * i_geo))
    result: dict[str, object] = {
        "pinion_torque_nm": torque, "tangential_force_n": wt, "pitch_line_velocity_m_s": v,
        "velocity_factor": kv, "lewis_factor_pinion": y1, "lewis_factor_gear": y2,
        "bending_stress_pinion_pa": sb1, "bending_stress_gear_pa": sb2, "contact_stress_pa": sc,
        "elastic_coefficient_sqrt_pa": cp, "face_width_to_module": f / m}
    if allowable_bending_pa is not None:
        sa = _finite_positive("allowable_bending_pa", allowable_bending_pa)
        result["bending_factor_of_safety_pinion"] = sa / sb1
        result["bending_factor_of_safety_gear"] = sa / sb2
    if allowable_contact_pa is not None:
        result["contact_factor_of_safety"] = _finite_positive("allowable_contact_pa", allowable_contact_pa) / sc
    return result


def spur_gear_module_for_bending(power_w: float, pinion_rpm: float, pinion_teeth: int,
                                 allowable_bending_pa: float, *, face_width_to_module: float = 10.0,
                                 finish: str = "hobbed", application_factor: float = 1.0) -> dict[str, float]:
    """Smallest module for which the pinion Lewis bending stress equals the allowable (bisection), with F = k*m.

    ``recommended_standard_module_m`` is the next preferred module (ISO 54 series) at or above the result.
    """
    p = _finite_positive("power_w", power_w)
    n = _finite_positive("pinion_rpm", pinion_rpm)
    n1 = _teeth("pinion_teeth", pinion_teeth)
    sa = _finite_positive("allowable_bending_pa", allowable_bending_pa)
    k = _finite_positive("face_width_to_module", face_width_to_module)
    ko = _finite_positive("application_factor", application_factor)
    y = lewis_form_factor(n1)
    omega = 2 * math.pi * n / 60
    torque = p / omega

    def stress(m: float) -> float:
        d = m * n1
        return 2 * torque / d * ko * _velocity_factor(omega * d / 2, finish) / (k * m * m * y)

    lo, hi = 1e-6, 10.0
    if stress(hi) > sa:
        raise ValueError("No module up to 10 m satisfies the allowable stress; check the inputs.")
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if stress(mid) > sa else (lo, mid)
    m = hi
    std = next((s / 1000 for s in STANDARD_MODULES_MM if s / 1000 >= m * (1 - 1e-9)), math.nan)
    return {"module_m": m, "face_width_m": k * m, "pinion_pitch_diameter_m": m * n1,
            "recommended_standard_module_m": std}


def _ask_float(prompt: str) -> float:
    try:
        value = float(input(prompt))
    except ValueError as exc:
        raise ValueError("Please enter a valid number.") from exc
    if not math.isfinite(value):
        raise ValueError("Please enter a finite number.")
    return value


def _ask_optional_ei() -> tuple:
    e_pa = _ask_float("Young's modulus E (Pa, 0 to skip deflection): ")
    if e_pa <= 0:
        return ()
    return (e_pa, _ask_float("Second moment I (m^4): "))


def _print_beam_result(r: dict) -> None:
    print(f"Reactions: left {r['reaction_left_n']:.6g} N, right {r['reaction_right_n']:.6g} N")
    print(f"End moments (sagging +): left {r['moment_at_left_nm']:.6g} N·m, right {r['moment_at_right_nm']:.6g} N·m")
    print(f"Maximum |moment|: {r['maximum_moment_nm']:.6g} N·m at x = {r['maximum_moment_location_m']:.6g} m")
    print(f"Maximum |shear|: {r['maximum_shear_n']:.6g} N")
    if "maximum_deflection_m" in r:
        print(f"Maximum deflection: {abs(r['maximum_deflection_m']):.6g} m at x = {r['maximum_deflection_location_m']:.6g} m")


def main() -> None:
    menu = """
Mechanical Engineering Toolkit
1. Linear thermal expansion
2. Shaft torque from power and RPM
3. Spring-mass natural frequency
4. Factor of safety
5. Simply supported beam with one point load
6. Unit conversion
7. Reynolds number
8. Pump power
9. Heat conduction through a wall
10. Gear-pair speed and torque
11. Heat-engine efficiency
12. Rectangular section properties
13. Beam deflection (point load)
14. Euler column buckling
15. Von Mises stress
16. Solid shaft torsion / sizing
17. Thin-wall pressure vessel
18. Carnot efficiency
19. Simply supported beam, uniform load
20. Cantilever, end point load
21. Cantilever, uniform load
22. Multi-load beam (point + distributed loads)
23. Fixed-fixed beam
24. Fatigue: modified Goodman check
25. Heat exchanger: LMTD method
26. Heat exchanger: effectiveness-NTU
27. Stress concentration: Kt, notch sensitivity and Kf
28. Fatigue: finite life from the S-N curve
29. Fatigue: Miner cumulative damage
30. Continuous (multi-span) beam
31. Pipe flow: Darcy-Weisbach pressure drop
32. Rolling-bearing life (L10)
33. Bolted joint: preload, torque and safety factors
34. Straight fin heat transfer
35. Internal-flow convection coefficient (Nusselt)
36. Circular section properties (solid / hollow)
37. Radiation heat transfer
38. Damped vibration and isolation
39. Spur gear pair geometry
40. Spur gear bending and contact stress
41. Spur gear module sizing (bending)
0. Exit
Select an option: """
    while True:
        choice = input(menu).strip()
        try:
            if choice == "0":
                print("Goodbye!")
                return
            if choice == "1":
                r = thermal_expansion(_ask_float("Length (m): "), _ask_float("Expansion coefficient (1/°C): "), _ask_float("Temperature change (°C): "))
                print(f"Length change: {r['delta_length_m']:.6g} m\nFinal length: {r['final_length_m']:.6g} m")
            elif choice == "2":
                r = shaft_torque(_ask_float("Power (W): "), _ask_float("Rotational speed (RPM): "))
                print(f"Angular speed: {r['angular_speed_rad_s']:.6g} rad/s\nTorque: {r['torque_nm']:.6g} N·m")
            elif choice == "3":
                r = spring_mass_frequency(_ask_float("Spring stiffness (N/m): "), _ask_float("Mass (kg): "))
                print(f"Natural frequency: {r['frequency_hz']:.6g} Hz\nPeriod: {r['period_s']:.6g} s")
            elif choice == "4":
                print(f"Factor of safety: {factor_of_safety(_ask_float('Failure strength (Pa): '), _ask_float('Working stress (Pa): ')):.6g}")
            elif choice == "5":
                r = simply_supported_beam_point_load(_ask_float("Span (m): "), _ask_float("Point load (N): "), _ask_float("Load position from left (m): "))
                print(f"Left reaction: {r['reaction_left_n']:.6g} N\nRight reaction: {r['reaction_right_n']:.6g} N\nMaximum moment: {r['maximum_moment_nm']:.6g} N·m")
                try:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                    axes[0].plot(r["x_m"], r["shear_n"]); axes[0].set_ylabel("Shear (N)"); axes[0].grid(True, alpha=.3)
                    axes[1].plot(r["x_m"], r["moment_nm"]); axes[1].set_ylabel("Moment (N·m)"); axes[1].set_xlabel("Position (m)"); axes[1].grid(True, alpha=.3)
                    fig.suptitle("Simply supported beam: point load"); fig.tight_layout(); plt.show()
                except ImportError:
                    print("Install matplotlib to plot the diagrams.")
            elif choice == "6":
                print(f"Converted value: {convert_units(_ask_float('Value: '), input('From unit: ').strip(), input('To unit: ').strip()):.8g}")
            elif choice == "7":
                re = reynolds_number(_ask_float("Fluid density (kg/m^3): "), _ask_float("Flow speed (m/s): "), _ask_float("Characteristic diameter (m): "), _ask_float("Dynamic viscosity (Pa·s): "))
                regime = "typically laminar" if re < 2300 else "typically transitional" if re < 4000 else "typically turbulent"
                print(f"Reynolds number: {re:.6g} ({regime} for pipe flow; thresholds are approximate)")
            elif choice == "8":
                r = pump_power(_ask_float("Flow rate (m^3/s): "), _ask_float("Pressure rise (Pa): "), _ask_float("Efficiency (fraction, e.g. 0.75): "))
                print(f"Hydraulic power: {r['hydraulic_power_w']:.6g} W\nInput power: {r['input_power_w']:.6g} W")
            elif choice == "9":
                q = conduction_heat_rate(_ask_float("Conductivity k (W/m·K): "), _ask_float("Area (m^2): "), _ask_float("Hot-side temperature (°C): "), _ask_float("Cold-side temperature (°C): "), _ask_float("Wall thickness (m): "))
                print(f"Steady conduction heat-transfer rate: {q:.6g} W")
            elif choice == "10":
                r = gear_train_output(_ask_float("Input speed (RPM): "), int(_ask_float("Driver gear teeth: ")), int(_ask_float("Driven gear teeth: ")), _ask_float("Input torque (N·m): "), _ask_float("Efficiency (fraction, e.g. 0.95): "))
                print(f"Gear ratio (driven/driver): {r['gear_ratio']:.6g}:1\nOutput speed: {r['output_rpm']:.6g} RPM\nOutput torque: {r['output_torque_nm']:.6g} N·m")
            elif choice == "11":
                r = heat_engine_performance(_ask_float("Heat input rate (W): "), _ask_float("Heat rejected rate (W): "))
                print(f"Net power: {r['net_power_w']:.6g} W\nThermal efficiency: {r['thermal_efficiency_percent']:.6g}%")
            elif choice == "12":
                r = rectangular_section_properties(_ask_float("Width b (m): "), _ask_float("Height h (m): "))
                print(f"Area: {r['area_m2']:.6g} m^2\nI: {r['second_moment_m4']:.6g} m^4\nSection modulus S: {r['section_modulus_m3']:.6g} m^3")
            elif choice == "13":
                r = beam_deflection_point_load(_ask_float("Span (m): "), _ask_float("Point load (N): "), _ask_float("Load position from left (m): "), _ask_float("Young's modulus E (Pa): "), _ask_float("Second moment I (m^4): "))
                print(f"Maximum deflection: {r['maximum_deflection_m']:.6g} m at x = {r['maximum_deflection_location_m']:.6g} m\nMidspan deflection: {r['midspan_deflection_m']:.6g} m")
            elif choice == "14":
                r = euler_buckling_load(_ask_float("Young's modulus E (Pa): "), _ask_float("Second moment I (m^4): "), _ask_float("Column length (m): "), _ask_float("Effective length factor K (1.0 pinned-pinned): "))
                print(f"Critical buckling load: {r['critical_load_n']:.6g} N (effective length {r['effective_length_m']:.6g} m)")
            elif choice == "15":
                print(f"Von Mises stress: {von_mises_stress(_ask_float('sigma_x: '), _ask_float('sigma_y: '), _ask_float('tau_xy: ')):.6g} (same units as input)")
            elif choice == "16":
                torque = _ask_float("Torque (N·m): ")
                mode = input("1 = stress from diameter, 2 = diameter from allowable shear: ").strip()
                if mode == "2":
                    print(f"Minimum diameter: {shaft_diameter_for_torque(torque, _ask_float('Allowable shear stress (Pa): ')):.6g} m")
                else:
                    print(f"Max shear stress: {torsion_solid_shaft(torque, _ask_float('Diameter (m): '))['max_shear_stress_pa']:.6g} Pa")
            elif choice == "17":
                r = thin_wall_pressure_vessel(_ask_float("Gauge pressure (Pa): "), _ask_float("Mean radius (m): "), _ask_float("Wall thickness (m): "))
                note = "" if r["thin_wall_valid"] else "  (WARNING: r/t < 10, thin-wall theory not reliable)"
                print(f"Cylinder hoop: {r['cylinder_hoop_stress_pa']:.6g} Pa\nCylinder longitudinal: {r['cylinder_longitudinal_stress_pa']:.6g} Pa\nSphere: {r['sphere_stress_pa']:.6g} Pa{note}")
            elif choice == "18":
                print(f"Carnot efficiency: {100 * carnot_efficiency(_ask_float('Hot reservoir (K): '), _ask_float('Cold reservoir (K): ')):.6g}%")
            elif choice in ("19", "20", "21"):
                span = _ask_float("Span/length (m): ")
                load = _ask_float("Point load (N): " if choice == "20" else "Uniform load (N/m): ")
                e_pa = _ask_float("Young's modulus E (Pa, 0 to skip deflection): ")
                i_m4 = _ask_float("Second moment I (m^4, 0 to skip deflection): ") if e_pa > 0 else None
                ei_args = (e_pa, i_m4) if e_pa > 0 else ()
                fn = {"19": simply_supported_beam_udl, "20": cantilever_beam_end_load, "21": cantilever_beam_udl}[choice]
                r = fn(span, load, *ei_args)
                print(f"Maximum |moment|: {r['maximum_moment_nm']:.6g} N·m")
                if "reaction_force_n" in r:
                    print(f"Fixed-end reaction: {r['reaction_force_n']:.6g} N, fixed-end moment: {r['fixed_end_moment_nm']:.6g} N·m")
                else:
                    print(f"Reactions: {r['reaction_left_n']:.6g} N / {r['reaction_right_n']:.6g} N")
                if "maximum_deflection_m" in r:
                    print(f"Maximum deflection: {r['maximum_deflection_m']:.6g} m")
            elif choice == "22":
                support = {"1": "simply_supported", "2": "cantilever", "3": "fixed_fixed", "4": "propped_cantilever"}.get(
                    input("Support: 1 = simply supported, 2 = cantilever, 3 = fixed-fixed, 4 = propped cantilever (fixed left, pin right): ").strip())
                if support is None:
                    raise ValueError("Choose 1, 2, 3 or 4.")
                span = _ask_float("Span/length (m): ")
                points = parse_numeric_rows(input("Point loads as 'position, load' rows separated by ';' (blank for none): "), 2)
                spreads = parse_numeric_rows(input("Uniform loads as 'start, end, w(N/m)' rows separated by ';' (blank for none): "), 3)
                _print_beam_result(beam_multi_load(span, points, spreads, support, *_ask_optional_ei()))
            elif choice == "23":
                span = _ask_float("Span (m): ")
                kind = input("1 = point load, 2 = uniform load over the full span: ").strip()
                if kind == "1":
                    load, pos = _ask_float("Point load (N): "), _ask_float("Load position from left (m): ")
                    r = fixed_fixed_beam_point_load(span, load, pos, *_ask_optional_ei())
                elif kind == "2":
                    r = fixed_fixed_beam_udl(span, _ask_float("Uniform load (N/m): "), *_ask_optional_ei())
                else:
                    raise ValueError("Choose 1 or 2.")
                _print_beam_result(r)
            elif choice == "24":
                sut = _ask_float("Ultimate tensile strength Sut (Pa): ")
                se = _ask_float("Corrected endurance limit Se (Pa): ")
                sy = _ask_float("Yield strength Sy (Pa, 0 to skip): ")
                mode = input("1 = enter max/min stress, 2 = enter mean/amplitude: ").strip()
                if mode == "2":
                    cycle = {"stress_mean_pa": _ask_float("Mean stress (Pa): "), "stress_amplitude_pa": _ask_float("Alternating amplitude (Pa): ")}
                else:
                    cycle = {"stress_max_pa": _ask_float("Maximum stress (Pa): "), "stress_min_pa": _ask_float("Minimum stress (Pa): ")}
                r = goodman_fatigue(sut, se, yield_strength_pa=sy if sy > 0 else None, **cycle)
                print(f"Mean stress: {r['stress_mean_pa']:.6g} Pa\nAlternating stress: {r['stress_amplitude_pa']:.6g} Pa")
                print(f"Goodman fatigue factor of safety: {r['goodman_factor_of_safety']:.6g} ({'infinite life predicted' if r['infinite_life_predicted'] else 'fatigue failure predicted'})")
                if "yield_factor_of_safety" in r:
                    print(f"Yield factor of safety: {r['yield_factor_of_safety']:.6g}")
            elif choice == "25":
                temps = [_ask_float(label) for label in ("Hot inlet (°C): ", "Hot outlet (°C): ", "Cold inlet (°C): ", "Cold outlet (°C): ")]
                arrangement = "parallel_flow" if input("1 = counterflow, 2 = parallel flow: ").strip() == "2" else "counterflow"
                f = _ask_float("Correction factor F (1 for pure counter/parallel flow): ")
                u = _ask_float("Overall U (W/m²·K, 0 to skip): ")
                q = _ask_float("Heat duty (W, 0 to skip): ")
                kwargs = {}
                if u > 0 and q > 0:
                    kwargs = {"overall_u_w_m2k": u, "heat_duty_w": q}
                elif u > 0:
                    kwargs = {"overall_u_w_m2k": u, "area_m2": _ask_float("Area (m²): ")}
                r = heat_exchanger_lmtd(*temps, arrangement=arrangement, correction_factor=f, **kwargs)
                print(f"dT1: {r['delta_t1_k']:.6g} K, dT2: {r['delta_t2_k']:.6g} K\nLMTD: {r['lmtd_k']:.6g} K (effective with F: {r['effective_lmtd_k']:.6g} K)")
                if "heat_duty_w" in r:
                    print(f"Heat duty: {r['heat_duty_w']:.6g} W")
                if "required_area_m2" in r:
                    print(f"Required area: {r['required_area_m2']:.6g} m²")
            elif choice == "26":
                ch, cc = _ask_float("Hot-stream capacity rate m·cp (W/K): "), _ask_float("Cold-stream capacity rate m·cp (W/K): ")
                th, tc = _ask_float("Hot inlet (°C): "), _ask_float("Cold inlet (°C): ")
                names = NTU_ARRANGEMENTS
                for i, name in enumerate(names, 1):
                    print(f"  {i} = {name}")
                idx = int(_ask_float("Arrangement number: "))
                if not 1 <= idx <= len(names):
                    raise ValueError("Choose one of the listed arrangements.")
                ua = _ask_float("UA (W/K, 0 to enter a target effectiveness instead): ")
                if ua > 0:
                    r = heat_exchanger_ntu(ch, cc, th, tc, ua_w_k=ua, arrangement=names[idx - 1])
                else:
                    r = heat_exchanger_ntu(ch, cc, th, tc, effectiveness=_ask_float("Target effectiveness (0-1): "), arrangement=names[idx - 1])
                print(f"NTU: {r['ntu']:.6g}, UA: {r['ua_w_k']:.6g} W/K, Cr: {r['capacity_ratio']:.6g}\nEffectiveness: {r['effectiveness']:.6g}\nHeat rate: {r['heat_rate_w']:.6g} W")
                print(f"Hot outlet: {r['hot_outlet_c']:.6g} °C, cold outlet: {r['cold_outlet_c']:.6g} °C")
            elif choice == "27":
                kind = input("1 = circular hole in finite-width plate, 2 = elliptical hole in wide plate, 3 = I already know Kt: ").strip()
                if kind == "1":
                    r = stress_concentration_factor("circular_hole_finite_plate", width_m=_ask_float("Plate width w (m): "), hole_diameter_m=_ask_float("Hole diameter d (m): "))
                    kt, radius = r["kt"], _ask_float("Notch radius for Kf (m, = d/2 for a hole; 0 to skip Kf): ")
                elif kind == "2":
                    a_t, a_p = _ask_float("Semi-axis across the load a (m): "), _ask_float("Semi-axis along the load b (m): ")
                    r = stress_concentration_factor("elliptical_hole_infinite_plate", semi_axis_transverse_m=a_t, semi_axis_parallel_m=a_p)
                    kt, radius = r["kt"], _ask_float("Notch tip radius for Kf (m, b^2/a at the tip; 0 to skip Kf): ")
                elif kind == "3":
                    kt, radius = _ask_float("Kt: "), _ask_float("Notch radius for Kf (m, 0 to skip Kf): ")
                else:
                    raise ValueError("Choose 1, 2 or 3.")
                print(f"Kt: {kt:.6g}")
                if radius > 0:
                    sut = _ask_float("Ultimate tensile strength Sut (Pa): ")
                    loading = {"1": "bending", "2": "axial", "3": "torsion"}.get(input("Loading: 1 = bending, 2 = axial, 3 = torsion: ").strip())
                    if loading is None:
                        raise ValueError("Choose 1, 2 or 3.")
                    r = fatigue_notch_factor(kt, radius, sut, loading=loading)
                    print(f"Notch sensitivity q: {r['q']:.6g}\nFatigue notch factor Kf: {r['kf']:.6g}")
            elif choice == "28":
                sut = _ask_float("Ultimate tensile strength Sut (Pa): ")
                se = _ask_float("Corrected endurance limit Se (Pa): ")
                kf = _ask_float("Fatigue notch factor Kf (1 for none): ")
                target = _ask_float("Target life in cycles (0 to skip, otherwise at least 1000): ")
                mode = input("1 = enter max/min stress, 2 = enter mean/amplitude: ").strip()
                if mode == "2":
                    cycle = {"stress_mean_pa": _ask_float("Mean stress (Pa): "), "stress_amplitude_pa": _ask_float("Alternating amplitude (Pa): ")}
                else:
                    cycle = {"stress_max_pa": _ask_float("Maximum stress (Pa): "), "stress_min_pa": _ask_float("Minimum stress (Pa): ")}
                r = sn_fatigue_life(sut, se, notch_factor_kf=kf, target_cycles=target if target > 0 else None, **cycle)
                print(f"Equivalent fully reversed stress: {r['equivalent_reversed_stress_pa']:.6g} Pa")
                if r["infinite_life_predicted"]:
                    print("Predicted life: infinite (stress is at or below the endurance limit)")
                else:
                    print(f"Predicted life: {r['life_cycles']:.6g} cycles")
                    if not r["high_cycle_range_valid"]:
                        print("WARNING: life is below 1e3 cycles; the high-cycle S-N model is not valid there.")
                if "factor_of_safety_at_target" in r:
                    print(f"Strength at {r['target_cycles']:.6g} cycles: {r['fatigue_strength_at_target_pa']:.6g} Pa, factor of safety {r['factor_of_safety_at_target']:.6g}")
                    print(f"Largest nominal amplitude allowed at that life: {r['allowable_amplitude_at_target_pa']:.6g} Pa")
            elif choice == "29":
                sut, se = _ask_float("Ultimate tensile strength Sut (Pa): "), _ask_float("Corrected endurance limit Se (Pa): ")
                kf = _ask_float("Fatigue notch factor Kf (1 for none): ")
                rows = parse_numeric_rows(input("Blocks as 'mean, amplitude, cycles' rows separated by ';' (Pa, Pa, cycles): "), 3)
                r = miner_cumulative_damage(sut, se, rows, notch_factor_kf=kf)
                for i, b in enumerate(r["blocks"], 1):
                    print(f"Block {i}: life {b['life_cycles']:.6g} cycles, damage {b['damage']:.6g}")
                print(f"Total damage D: {r['total_damage']:.6g} ({'failure predicted' if r['failure_predicted'] else 'no failure predicted'}); repeats to failure: {r['blocks_to_failure']:.6g}")
                if not r["high_cycle_range_valid"]:
                    print("WARNING: a block has life below 1e3 cycles; the high-cycle S-N model is not valid there.")
            elif choice == "30":
                spans = [row[0] for row in parse_numeric_rows(input("Span lengths in m, separated by ';' (e.g. 4; 5; 4): "), 1)]
                ends = {"1": "pinned", "2": "fixed", "3": "free"}
                left = ends.get(input("Left end: 1 = pinned, 2 = fixed, 3 = free (overhang): ").strip())
                right = ends.get(input("Right end: 1 = pinned, 2 = fixed, 3 = free (overhang): ").strip())
                if left is None or right is None:
                    raise ValueError("Choose 1, 2 or 3 for each end.")
                points = parse_numeric_rows(input("Point loads as 'position, load' rows separated by ';' (blank for none): "), 2)
                spreads = parse_numeric_rows(input("Uniform loads as 'start, end, w(N/m)' rows separated by ';' (blank for none): "), 3)
                e_pa = _ask_float("Young's modulus E (Pa, 0 to skip deflection): ")
                if e_pa > 0:
                    r = continuous_beam(spans, points, spreads, left_end=left, right_end=right, youngs_modulus_pa=e_pa, second_moment_m4=_ask_float("Second moment I (m^4): "))
                else:
                    r = continuous_beam(spans, points, spreads, left_end=left, right_end=right)
                for x, rx, m in zip(r["support_positions_m"], r["reactions_n"], r["moments_at_supports_nm"]):
                    print(f"Support at x = {x:.6g} m: reaction {rx:.6g} N (upward +), moment {m:.6g} N·m")
                print(f"Largest sagging moment: {r['maximum_sagging_moment_nm']:.6g} N·m at x = {r['maximum_sagging_moment_location_m']:.6g} m")
                print(f"Largest hogging moment: {r['maximum_hogging_moment_nm']:.6g} N·m at x = {r['maximum_hogging_moment_location_m']:.6g} m")
                print(f"Maximum |shear|: {r['maximum_shear_n']:.6g} N")
                if "maximum_deflection_m" in r:
                    print(f"Maximum deflection: {abs(r['maximum_deflection_m']):.6g} m at x = {r['maximum_deflection_location_m']:.6g} m")
            elif choice == "31":
                d, length = _ask_float("Inside diameter (m): "), _ask_float("Pipe length (m): ")
                rho, mu = _ask_float("Fluid density (kg/m^3): "), _ask_float("Dynamic viscosity (Pa·s): ")
                if input("1 = enter flow rate, 2 = enter velocity: ").strip() == "2":
                    flow = {"velocity_m_s": _ask_float("Mean velocity (m/s): ")}
                else:
                    flow = {"flow_rate_m3_s": _ask_float("Flow rate (m^3/s): ")}
                r = pipe_pressure_drop(d, length, rho, mu, roughness_m=_ask_float("Wall roughness (m, 4.5e-5 for commercial steel, 0 for smooth): "),
                                       minor_loss_k=_ask_float("Sum of minor-loss coefficients K (0 for none): "),
                                       elevation_gain_m=_ask_float("Elevation gain inlet to outlet (m, negative if downhill): "), **flow)
                print(f"Velocity: {r['velocity_m_s']:.6g} m/s, Reynolds number: {r['reynolds_number']:.6g} ({r['flow_regime']})")
                print(f"Friction factor: {r['friction_factor']:.6g}\nFriction loss: {r['friction_pressure_drop_pa']:.6g} Pa ({r['friction_head_loss_m']:.6g} m)")
                print(f"Minor loss: {r['minor_pressure_drop_pa']:.6g} Pa\nElevation: {r['elevation_pressure_change_pa']:.6g} Pa")
                print(f"Total pressure drop: {r['total_pressure_drop_pa']:.6g} Pa ({r['total_head_m']:.6g} m of fluid)\nHydraulic power: {r['hydraulic_power_w']:.6g} W")
            elif choice == "32":
                c_rating = _ask_float("Basic dynamic load rating C (N): ")
                p_load = _ask_float("Equivalent dynamic load P (N, or 0 to build it from radial and axial loads): ")
                if p_load <= 0:
                    p_load = bearing_equivalent_load(_ask_float("Radial load Fr (N): "), _ask_float("Axial load Fa (N): "),
                                                     radial_factor_x=_ask_float("Radial factor X: "), axial_factor_y=_ask_float("Axial factor Y: "))
                    print(f"Equivalent load P: {p_load:.6g} N")
                kind = "roller" if input("1 = ball bearing, 2 = roller bearing: ").strip() == "2" else "ball"
                rpm = _ask_float("Speed (RPM, 0 to skip hours): ")
                target = _ask_float("Target life (hours, 0 to skip; needs a speed): ")
                r = bearing_life_l10(c_rating, p_load, speed_rpm=rpm if rpm > 0 else None, bearing_type=kind,
                                     reliability_percent=_ask_float("Reliability (%, 90 to 99): "),
                                     target_life_hours=target if target > 0 else None)
                print(f"L10: {r['l10_million_revolutions']:.6g} million revolutions (a1 = {r['reliability_factor_a1']:.3g}, adjusted {r['adjusted_life_million_revolutions']:.6g})")
                if "l10_hours" in r:
                    print(f"L10: {r['l10_hours']:.6g} h, adjusted for reliability: {r['adjusted_life_hours']:.6g} h")
                if "required_dynamic_rating_n" in r:
                    print(f"Required dynamic rating for the target life: {r['required_dynamic_rating_n']:.6g} N (margin {r['rating_margin']:.3g}x)")
            elif choice == "33":
                d = _ask_float("Nominal bolt diameter (m): ")
                at = _ask_float("Tensile-stress area (m^2, or 0 to compute it from the thread pitch): ")
                if at <= 0:
                    at = metric_tensile_stress_area(d, _ask_float("Thread pitch (m): "))
                    print(f"Tensile-stress area: {at:.6g} m^2")
                sp = _ask_float("Proof strength Sp (Pa): ")
                c = _ask_float("Joint constant C = kb/(kb+km) (0-1): ")
                p = _ask_float("External tensile load on the joint (N): ")
                fi = _ask_float("Preload (N, 0 to use a fraction of the proof load): ")
                frac = 0.75 if fi > 0 else _ask_float("Preload fraction of proof load (0.75 reusable, 0.9 permanent): ")
                r = bolted_joint(d, at, sp, c, external_load_n=p, preload_fraction=frac if fi <= 0 else 0.75,
                                 preload_n=fi if fi > 0 else None, torque_coefficient_k=_ask_float("Torque coefficient K (0.2 typical): "))
                print(f"Proof load: {r['proof_load_n']:.6g} N, preload: {r['preload_n']:.6g} N ({100 * r['preload_fraction']:.3g}% of proof)")
                print(f"Tightening torque: {r['tightening_torque_nm']:.6g} N·m\nBolt load: {r['bolt_load_n']:.6g} N (stress {r['bolt_stress_pa']:.6g} Pa)\nClamp load: {r['clamp_load_n']:.6g} N")
                print(f"Proof factor: {r['proof_factor_of_safety']:.6g}, load factor: {r['load_factor']:.6g}, separation factor: {r['separation_factor']:.6g}")
                if not r["joint_stays_clamped"]:
                    print("WARNING: the joint separates under this load.")
            elif choice == "34":
                tip = {"1": "adiabatic", "2": "convective", "3": "infinite"}.get(input("Tip: 1 = adiabatic, 2 = convective, 3 = infinitely long: ").strip())
                if tip is None:
                    raise ValueError("Choose 1, 2 or 3.")
                r = straight_fin(_ask_float("Fin conductivity k (W/m·K): "), _ask_float("Convection coefficient h (W/m^2·K): "), _ask_float("Fin length (m): "),
                                 _ask_float("Fin thickness (m): "), _ask_float("Fin width (m): "), _ask_float("Base temperature (°C): "), _ask_float("Ambient temperature (°C): "), tip=tip)
                print(f"m = {r['fin_parameter_m_per_m']:.6g} 1/m, mL = {r['ml']:.6g}\nFin heat rate: {r['heat_rate_w']:.6g} W\nTip temperature: {r['tip_temperature_c']:.6g} °C")
                print(f"Fin efficiency: {r['efficiency']:.6g}, effectiveness: {r['effectiveness']:.6g}")
                if not r["thin_fin_valid"]:
                    print(f"WARNING: Biot number {r['biot_number']:.3g} > 0.1, the 1-D fin model is not reliable.")
            elif choice == "35":
                re, pr = _ask_float("Reynolds number: "), _ask_float("Prandtl number: ")
                d, k = _ask_float("Tube diameter (m): "), _ask_float("Fluid conductivity (W/m·K): ")
                corr = "dittus_boelter" if input("1 = Gnielinski, 2 = Dittus-Boelter: ").strip() == "2" else "gnielinski"
                heating = input("1 = fluid is being heated, 2 = cooled: ").strip() != "2"
                wall = "constant_heat_flux" if input("Laminar wall condition: 1 = constant temperature, 2 = constant heat flux: ").strip() == "2" else "constant_temperature"
                r = convection_internal_flow(re, pr, d, k, correlation=corr, heating=heating, wall_condition=wall)
                print(f"{r['flow_regime'].capitalize()} flow ({r['correlation']}): Nu = {r['nusselt_number']:.6g}, h = {r['heat_transfer_coefficient_w_m2k']:.6g} W/m²·K")
                if not r["valid_range"]:
                    print("WARNING: outside the validity range of this correlation; treat the result as a rough estimate.")
            elif choice == "36":
                r = circular_section_properties(_ask_float("Outer diameter (m): "), _ask_float("Inner diameter (m, 0 for solid): "))
                print(f"Area: {r['area_m2']:.6g} m^2\nI: {r['second_moment_m4']:.6g} m^4, J: {r['polar_moment_m4']:.6g} m^4\nSection modulus S: {r['section_modulus_m3']:.6g} m^3, plastic Z: {r['plastic_modulus_m3']:.6g} m^3\nRadius of gyration: {r['radius_of_gyration_m']:.6g} m")
            elif choice == "37":
                r = radiation_heat_rate(_ask_float("Emissivity (0-1): "), _ask_float("Surface area (m^2): "), _ask_float("Surface temperature (K): "), _ask_float("Surroundings temperature (K): "))
                print(f"Net radiation heat rate: {r['heat_rate_w']:.6g} W (positive = surface loses heat)\nRadiation coefficient h_rad: {r['radiation_coefficient_w_m2k']:.6g} W/m²·K")
            elif choice == "38":
                freq = _ask_float("Forcing frequency (Hz, 0 to skip): ")
                r = damped_vibration(_ask_float("Mass (kg): "), _ask_float("Stiffness (N/m): "), _ask_float("Damping coefficient c (N·s/m, 0 for none): "),
                                     forcing_frequency_hz=freq if freq > 0 else None)
                print(f"Natural frequency: {r['natural_frequency_hz']:.6g} Hz, damping ratio: {r['damping_ratio']:.6g} ({r['damping_type']})")
                print(f"Critical damping: {r['critical_damping_n_s_m']:.6g} N·s/m, damped frequency: {r['damped_frequency_hz']:.6g} Hz")
                if r["logarithmic_decrement"] is not None:
                    print(f"Logarithmic decrement: {r['logarithmic_decrement']:.6g}")
                if "magnification_factor" in r:
                    print(f"Frequency ratio: {r['frequency_ratio']:.6g}, magnification: {r['magnification_factor']:.6g}, transmissibility: {r['transmissibility']:.6g}"
                          f" ({'isolates' if r['isolates_vibration'] else 'does not isolate'})")
            elif choice == "39":
                r = spur_gear_geometry(_ask_float("Module (m): "), int(_ask_float("Pinion teeth: ")), int(_ask_float("Gear teeth: ")), pressure_angle_deg=_ask_float("Pressure angle (degrees, 20 typical): "))
                print(f"Pitch diameters: {r['pinion_pitch_diameter_m']:.6g} / {r['gear_pitch_diameter_m']:.6g} m, centre distance {r['centre_distance_m']:.6g} m")
                print(f"Outside diameters: {r['pinion_outside_diameter_m']:.6g} / {r['gear_outside_diameter_m']:.6g} m\nRoot diameters: {r['pinion_root_diameter_m']:.6g} / {r['gear_root_diameter_m']:.6g} m")
                print(f"Speed ratio: {r['speed_ratio']:.6g}, contact ratio: {r['contact_ratio']:.4g}")
                if not r["undercut_free"]:
                    print(f"WARNING: pinion has fewer than {r['minimum_teeth_no_undercut']:.3g} teeth and will undercut without profile shift.")
            elif choice == "40":
                p_w, rpm, m = _ask_float("Power (W): "), _ask_float("Pinion speed (RPM): "), _ask_float("Module (m): ")
                n1, n2, fw = int(_ask_float("Pinion teeth: ")), int(_ask_float("Gear teeth: ")), _ask_float("Face width (m): ")
                sab, sac = _ask_float("Allowable bending stress (Pa, 0 to skip): "), _ask_float("Allowable contact stress (Pa, 0 to skip): ")
                r = spur_gear_stress(p_w, rpm, m, n1, n2, fw, application_factor=_ask_float("Application factor Ko (1 smooth, 1.25-2 shock): "),
                                     allowable_bending_pa=sab if sab > 0 else None, allowable_contact_pa=sac if sac > 0 else None)
                print(f"Torque: {r['pinion_torque_nm']:.6g} N·m, tangential force: {r['tangential_force_n']:.6g} N, pitch-line speed: {r['pitch_line_velocity_m_s']:.6g} m/s")
                print(f"Bending stress: pinion {r['bending_stress_pinion_pa']:.6g} Pa, gear {r['bending_stress_gear_pa']:.6g} Pa\nContact stress: {r['contact_stress_pa']:.6g} Pa")
                for key, label in (("bending_factor_of_safety_pinion", "Bending FoS pinion"), ("bending_factor_of_safety_gear", "Bending FoS gear"), ("contact_factor_of_safety", "Contact FoS")):
                    if key in r:
                        print(f"{label}: {r[key]:.4g}")
            elif choice == "41":
                r = spur_gear_module_for_bending(_ask_float("Power (W): "), _ask_float("Pinion speed (RPM): "), int(_ask_float("Pinion teeth: ")), _ask_float("Allowable bending stress (Pa): "), face_width_to_module=_ask_float("Face width / module (8-12 typical): "))
                print(f"Minimum module: {1000 * r['module_m']:.4g} mm (face width {1000 * r['face_width_m']:.4g} mm)\nNext standard module: {1000 * r['recommended_standard_module_m']:.4g} mm")
            else:
                print("Choose one of the listed options.")
        except ValueError as exc:
            print(f"Input error: {exc}")
        print()


if __name__ == "__main__":
    main()


# Machine-element design calculators (SI units)
def compression_spring_design(load_n: float, wire_diameter_m: float, mean_coil_diameter_m: float,
                              active_coils: float, shear_modulus_pa: float,
                              *, free_length_m: float | None = None) -> dict[str, float]:
    """Helical compression spring rate, deflection, Wahl-corrected shear stress and geometry."""
    P = _finite_positive("load_n", load_n, allow_zero=True)
    d = _finite_positive("wire_diameter_m", wire_diameter_m)
    D = _finite_positive("mean_coil_diameter_m", mean_coil_diameter_m)
    n = _finite_positive("active_coils", active_coils)
    G = _finite_positive("shear_modulus_pa", shear_modulus_pa)
    if D <= d:
        raise ValueError("Mean coil diameter must exceed wire diameter.")
    C = D / d
    if C <= 0.5:
        raise ValueError("Spring index D/d must be greater than 0.5.")
    wahl = (4*C - 1)/(4*C - 4) + 0.615/C
    rate = G*d**4/(8*D**3*n)
    delta = P/rate
    tau = wahl*8*P*D/(math.pi*d**3)
    result = {"spring_index": C, "wahl_factor": wahl, "spring_rate_n_per_m": rate,
              "deflection_m": delta, "maximum_shear_stress_pa": tau,
              "solid_length_m": (n+2)*d, "load_n": P}
    if free_length_m is not None:
        L0 = _finite_positive("free_length_m", free_length_m)
        result["remaining_length_at_load_m"] = L0-delta
        result["estimated_solid_clearance_m"] = L0-delta-(n+2)*d
        if delta >= L0:
            raise ValueError("Calculated deflection reaches or exceeds free length.")
    return result


def belt_drive_design(power_w: float, driver_diameter_m: float, driven_diameter_m: float,
                      speed_rpm: float, friction_coefficient: float, wrap_angle_rad: float) -> dict[str, float]:
    """Ideal flat-belt drive: speed ratio, tight/slack tensions and Euler belt friction."""
    P = _finite_positive("power_w", power_w, allow_zero=True)
    d1 = _finite_positive("driver_diameter_m", driver_diameter_m)
    d2 = _finite_positive("driven_diameter_m", driven_diameter_m)
    rpm = _finite_positive("speed_rpm", speed_rpm)
    mu = _finite_positive("friction_coefficient", friction_coefficient, allow_zero=True)
    theta = _finite_positive("wrap_angle_rad", wrap_angle_rad)
    v = math.pi*d1*rpm/60
    ratio = math.exp(min(700.0, mu*theta))
    diff = P/v if v else 0.0
    if ratio <= 1:
        if diff > 0: raise ValueError("Positive power requires positive friction and wrap angle.")
        t2 = t1 = 0.0
    else:
        t2 = diff/(ratio-1)
        t1 = ratio*t2
    return {"belt_speed_m_s": v, "driven_speed_rpm_ideal": rpm*d1/d2,
            "speed_ratio_driver_to_driven": d2/d1, "tension_ratio_max": ratio,
            "tight_side_tension_n": t1, "slack_side_tension_n": t2,
            "effective_tension_n": diff, "transmitted_power_w": P}


def power_screw_design(axial_load_n: float, mean_diameter_m: float, lead_m: float,
                       friction_coefficient: float, *, collar_friction_coefficient: float = 0.0,
                       collar_mean_diameter_m: float = 0.0) -> dict[str, float]:
    """Square-thread power screw raising/lowering torque, efficiency and self-locking check."""
    W = _finite_positive("axial_load_n", axial_load_n)
    dm = _finite_positive("mean_diameter_m", mean_diameter_m)
    lead = _finite_positive("lead_m", lead_m)
    mu = _finite_positive("friction_coefficient", friction_coefficient, allow_zero=True)
    muc = _finite_positive("collar_friction_coefficient", collar_friction_coefficient, allow_zero=True)
    dc = _finite_positive("collar_mean_diameter_m", collar_mean_diameter_m, allow_zero=True)
    if lead >= math.pi*dm:
        raise ValueError("Lead is too large for this mean diameter.")
    tan_lam = lead/(math.pi*dm)
    lam = math.atan(tan_lam)
    phi = math.atan(mu)
    raise_thread = W*dm/2 * (tan_lam+mu)/(1-mu*tan_lam)
    lower_thread = W*dm/2 * (mu-tan_lam)/(1+mu*tan_lam)
    collar = W*muc*dc/2
    torque_raise = raise_thread+collar
    torque_lower = lower_thread+collar
    turns = 2*math.pi/lead
    work_per_turn = W*lead
    return {"helix_angle_deg": math.degrees(lam), "friction_angle_deg": math.degrees(phi),
            "torque_to_raise_nm": torque_raise, "torque_to_lower_nm": torque_lower,
            "collar_friction_torque_nm": collar, "efficiency_raising": W*lead/(2*math.pi*torque_raise) if torque_raise else 0.0,
            "self_locking_predicted": mu > tan_lam, "ideal_turns_per_m": turns,
            "work_per_turn_j": work_per_turn}


def shaft_fatigue_design(bending_moment_nm: float, torque_mean_nm: float, torque_alternating_nm: float,
                         endurance_limit_pa: float, ultimate_strength_pa: float, yield_strength_pa: float,
                         *, factor_of_safety: float = 2.0, kf_bending: float = 1.0,
                         kf_torsion: float = 1.0) -> dict[str, float]:
    """Solid circular shaft sizing by von-Mises-equivalent modified Goodman plus yield check."""
    M = _finite_positive("bending_moment_nm", abs(float(bending_moment_nm)), allow_zero=True)
    Tm = _finite_positive("torque_mean_nm", abs(float(torque_mean_nm)), allow_zero=True)
    Ta = _finite_positive("torque_alternating_nm", abs(float(torque_alternating_nm)), allow_zero=True)
    Se = _finite_positive("endurance_limit_pa", endurance_limit_pa)
    Sut = _finite_positive("ultimate_strength_pa", ultimate_strength_pa)
    Sy = _finite_positive("yield_strength_pa", yield_strength_pa)
    n = _finite_positive("factor_of_safety", factor_of_safety)
    Kfb = _finite_positive("kf_bending", kf_bending)
    Kft = _finite_positive("kf_torsion", kf_torsion)
    # sigma_b = 32 M/(pi d^3); tau = 16 T/(pi d^3)
    # Equivalent alternating and mean stresses for a rotating shaft.
    A = math.sqrt((Kfb*32*M/math.pi)**2 + 3*(Kft*16*Ta/math.pi)**2)
    B = math.sqrt(3*(Kft*16*Tm/math.pi)**2)
    denom = (A/Se + B/Sut)
    d_goodman = (n*denom)**(1/3) if denom > 0 else 0.0
    # Conservative peak von Mises yield diameter using combined bending and total torque.
    Tpeak = Tm + Ta
    d_yield = (n*math.sqrt((32*M/math.pi)**2 + 3*(16*Tpeak/math.pi)**2)/Sy)**(1/3)
    d = max(d_goodman, d_yield)
    return {"required_diameter_m": d, "required_diameter_mm": d*1000,
            "goodman_diameter_mm": d_goodman*1000, "yield_check_diameter_mm": d_yield*1000,
            "factor_of_safety_target": n, "equivalent_alternating_stress_at_design_pa": A/d**3 if d else 0.0,
            "equivalent_mean_stress_at_design_pa": B/d**3 if d else 0.0,
            "assumptions": "Solid circular shaft; rotating bending treated as alternating; simplified modified Goodman and von Mises yield checks."}
