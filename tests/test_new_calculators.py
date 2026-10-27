import math
import pytest
from engineering_toolkit import (
    pipe_pressure_drop, bearing_equivalent_load, bearing_life_l10, metric_tensile_stress_area,
    bolted_joint, straight_fin, convection_internal_flow, circular_section_properties,
    radiation_heat_rate, damped_vibration,
)


# ---- pipe flow --------------------------------------------------------------------------------

def test_laminar_pipe_matches_hagen_poiseuille():
    r = pipe_pressure_drop(0.01, 1.0, 1000, 0.001, velocity_m_s=0.1)
    assert r["flow_regime"] == "laminar"
    assert r["reynolds_number"] == pytest.approx(1000)
    assert r["total_pressure_drop_pa"] == pytest.approx(32 * 0.001 * 1.0 * 0.1 / 0.01 ** 2)


def test_turbulent_pipe_satisfies_colebrook():
    r = pipe_pressure_drop(0.05, 20.0, 998, 1.0e-3, flow_rate_m3_s=0.004, roughness_m=4.5e-5)
    f, re, rr = r["friction_factor"], r["reynolds_number"], r["relative_roughness"]
    assert r["flow_regime"] == "turbulent"
    assert 1 / math.sqrt(f) == pytest.approx(-2 * math.log10(rr / 3.7 + 2.51 / (re * math.sqrt(f))), rel=1e-9)


def test_smooth_pipe_friction_factor_textbook_value():
    r = pipe_pressure_drop(0.1, 10, 1000, 1e-3, velocity_m_s=1.0, roughness_m=0)
    assert r["friction_factor"] == pytest.approx(0.01799, rel=2e-3)   # Re = 1e5, smooth pipe


def test_flow_rate_and_velocity_inputs_agree_and_losses_add():
    a = pipe_pressure_drop(0.05, 30, 1000, 1e-3, flow_rate_m3_s=0.002, minor_loss_k=2.5, elevation_gain_m=5)
    b = pipe_pressure_drop(0.05, 30, 1000, 1e-3, velocity_m_s=a["velocity_m_s"], minor_loss_k=2.5, elevation_gain_m=5)
    assert a["total_pressure_drop_pa"] == pytest.approx(b["total_pressure_drop_pa"])
    assert a["total_pressure_drop_pa"] == pytest.approx(a["friction_pressure_drop_pa"] + a["minor_pressure_drop_pa"] + a["elevation_pressure_change_pa"])
    assert a["elevation_pressure_change_pa"] == pytest.approx(1000 * 9.80665 * 5)
    assert a["hydraulic_power_w"] == pytest.approx(a["total_pressure_drop_pa"] * 0.002)


def test_pipe_zero_flow_and_bad_input():
    assert pipe_pressure_drop(0.05, 10, 1000, 1e-3, velocity_m_s=0)["friction_pressure_drop_pa"] == 0
    with pytest.raises(ValueError):
        pipe_pressure_drop(0.05, 10, 1000, 1e-3)
    with pytest.raises(ValueError):
        pipe_pressure_drop(0.05, 10, 1000, 1e-3, velocity_m_s=1, flow_rate_m3_s=1)
    with pytest.raises(ValueError):
        pipe_pressure_drop(0.05, 10, 1000, 1e-3, velocity_m_s=1, roughness_m=0.01)
    with pytest.raises(ValueError):
        pipe_pressure_drop(-0.05, 10, 1000, 1e-3, velocity_m_s=1)


# ---- bearings ---------------------------------------------------------------------------------

def test_bearing_l10_ball_and_roller():
    ball = bearing_life_l10(20_000, 10_000, speed_rpm=1000)
    assert ball["l10_million_revolutions"] == pytest.approx(8.0)
    assert ball["l10_hours"] == pytest.approx(8e6 / 60_000)
    assert bearing_life_l10(20_000, 10_000, bearing_type="roller")["l10_million_revolutions"] == pytest.approx(2 ** (10 / 3))


def test_bearing_reliability_factor_and_required_rating():
    r = bearing_life_l10(20_000, 10_000, speed_rpm=1000, reliability_percent=95)
    assert r["reliability_factor_a1"] == pytest.approx(0.64)
    assert r["adjusted_life_hours"] == pytest.approx(0.64 * r["l10_hours"])
    assert bearing_life_l10(20_000, 10_000, reliability_percent=97.5)["reliability_factor_a1"] == pytest.approx(0.42)
    need = bearing_life_l10(20_000, 10_000, speed_rpm=1000, target_life_hours=r["adjusted_life_hours"], reliability_percent=95)
    assert need["required_dynamic_rating_n"] == pytest.approx(20_000)
    assert need["rating_margin"] == pytest.approx(1.0)


def test_bearing_equivalent_load_and_errors():
    assert bearing_equivalent_load(5000, 2000, radial_factor_x=0.56, axial_factor_y=1.5) == pytest.approx(0.56 * 5000 + 1.5 * 2000)
    with pytest.raises(ValueError):
        bearing_equivalent_load(0, 0)
    with pytest.raises(ValueError):
        bearing_life_l10(1000, 500, reliability_percent=80)
    with pytest.raises(ValueError):
        bearing_life_l10(1000, 500, target_life_hours=1000)
    with pytest.raises(ValueError):
        bearing_life_l10(1000, 500, bearing_type="needle")


# ---- bolted joints ----------------------------------------------------------------------------

def test_metric_stress_area_m10():
    assert metric_tensile_stress_area(0.010, 0.0015) == pytest.approx(58.0e-6, rel=1e-3)


def test_bolted_joint_load_sharing_and_factors():
    at, sp, c, p = 58e-6, 580e6, 0.25, 8000.0
    r = bolted_joint(0.010, at, sp, c, external_load_n=p)
    fi = 0.75 * sp * at
    assert r["preload_n"] == pytest.approx(fi)
    assert r["tightening_torque_nm"] == pytest.approx(0.2 * fi * 0.010)
    assert r["bolt_load_n"] == pytest.approx(fi + c * p)
    assert r["bolt_load_n"] + 0 == pytest.approx(r["clamp_load_n"] + p)         # equilibrium: Fb = Fm + P
    assert r["separation_factor"] == pytest.approx(fi / ((1 - c) * p))
    assert r["load_factor"] == pytest.approx((sp * at - fi) / (c * p))
    assert r["proof_factor_of_safety"] == pytest.approx(sp * at / r["bolt_load_n"])
    assert r["joint_stays_clamped"]


def test_bolted_joint_no_load_separation_and_explicit_preload():
    r = bolted_joint(0.010, 58e-6, 580e6, 0.3)
    assert math.isinf(r["load_factor"]) and math.isinf(r["separation_factor"])
    sep = bolted_joint(0.010, 58e-6, 580e6, 0.3, external_load_n=50_000, preload_n=10_000)
    assert not sep["joint_stays_clamped"] and sep["separation_factor"] < 1
    with pytest.raises(ValueError):
        bolted_joint(0.010, 58e-6, 580e6, 0.3, preload_n=1e6)
    with pytest.raises(ValueError):
        bolted_joint(0.010, 58e-6, 580e6, 1.2)


# ---- fins -------------------------------------------------------------------------------------

FIN = dict(conductivity_w_mk=200.0, heat_transfer_coefficient_w_m2k=50.0, length_m=0.05, thickness_m=0.004,
           width_m=0.1, base_temperature_c=100.0, ambient_temperature_c=25.0)


def test_fin_adiabatic_formula():
    r = straight_fin(**FIN)
    h, k, w, t, L = 50.0, 200.0, 0.1, 0.004, 0.05
    p, a = 2 * (w + t), w * t
    m = math.sqrt(h * p / (k * a))
    assert r["heat_rate_w"] == pytest.approx(math.sqrt(h * p * k * a) * 75 * math.tanh(m * L))
    assert r["tip_temperature_c"] == pytest.approx(25 + 75 / math.cosh(m * L))
    assert 0 < r["efficiency"] <= 1 and r["effectiveness"] > 1


def test_fin_tip_conditions_order_and_long_fin_limit():
    ad = straight_fin(**FIN)["heat_rate_w"]
    cv = straight_fin(**FIN, tip="convective")["heat_rate_w"]
    inf = straight_fin(**FIN, tip="infinite")["heat_rate_w"]
    assert ad < cv < inf
    long_ = {**FIN, "length_m": 5.0}
    assert straight_fin(**long_)["heat_rate_w"] == pytest.approx(straight_fin(**long_, tip="infinite")["heat_rate_w"])
    assert straight_fin(**long_)["tip_temperature_c"] == pytest.approx(25.0, abs=1e-6)


def test_fin_cooling_sign_and_errors():
    assert straight_fin(**{**FIN, "base_temperature_c": 0.0})["heat_rate_w"] < 0
    assert straight_fin(**{**FIN, "base_temperature_c": 0.0})["efficiency"] > 0
    assert not straight_fin(**{**FIN, "conductivity_w_mk": 0.5})["thin_fin_valid"]
    with pytest.raises(ValueError):
        straight_fin(**{**FIN, "base_temperature_c": 25.0})
    with pytest.raises(ValueError):
        straight_fin(**FIN, tip="free")


# ---- convection -------------------------------------------------------------------------------

def test_nusselt_laminar_values():
    assert convection_internal_flow(1000, 5, 0.02, 0.6)["nusselt_number"] == pytest.approx(3.66)
    assert convection_internal_flow(1000, 5, 0.02, 0.6, wall_condition="constant_heat_flux")["nusselt_number"] == pytest.approx(4.36)
    assert convection_internal_flow(1000, 5, 0.02, 0.6)["heat_transfer_coefficient_w_m2k"] == pytest.approx(3.66 * 0.6 / 0.02)


def test_nusselt_turbulent_correlations():
    db = convection_internal_flow(1e5, 0.7, 0.05, 0.026, correlation="dittus_boelter")
    assert db["nusselt_number"] == pytest.approx(0.023 * 1e4 * 0.7 ** 0.4)
    assert convection_internal_flow(1e5, 0.7, 0.05, 0.026, correlation="dittus_boelter", heating=False)["nusselt_number"] > db["nusselt_number"]   # Pr < 1: 0.7^0.3 > 0.7^0.4
    gn = convection_internal_flow(1e5, 0.7, 0.05, 0.026)
    assert gn["nusselt_number"] == pytest.approx(178.6, rel=1e-2)
    assert gn["valid_range"] and gn["correlation"] == "Gnielinski"
    assert not convection_internal_flow(5000, 0.7, 0.05, 0.026, correlation="dittus_boelter")["valid_range"]
    with pytest.raises(ValueError):
        convection_internal_flow(1e5, 0.7, 0.05, 0.026, correlation="magic")


# ---- sections and radiation -------------------------------------------------------------------

def test_circular_sections():
    s = circular_section_properties(0.1)
    assert s["area_m2"] == pytest.approx(math.pi * 0.05 ** 2)
    assert s["second_moment_m4"] == pytest.approx(math.pi * 0.1 ** 4 / 64)
    assert s["polar_moment_m4"] == pytest.approx(2 * s["second_moment_m4"])
    assert s["radius_of_gyration_m"] == pytest.approx(0.1 / 4)
    h = circular_section_properties(0.1, 0.08)
    assert h["second_moment_m4"] == pytest.approx(math.pi * (0.1 ** 4 - 0.08 ** 4) / 64)
    assert h["plastic_modulus_m3"] == pytest.approx((0.1 ** 3 - 0.08 ** 3) / 6)
    with pytest.raises(ValueError):
        circular_section_properties(0.1, 0.1)


def test_radiation_heat_rate():
    r = radiation_heat_rate(1.0, 1.0, 400, 300)
    assert r["heat_rate_w"] == pytest.approx(5.670374419e-8 * (400 ** 4 - 300 ** 4))
    assert r["heat_rate_w"] == pytest.approx(r["radiation_coefficient_w_m2k"] * 100)
    assert radiation_heat_rate(0.5, 2.0, 300, 300)["heat_rate_w"] == 0
    assert radiation_heat_rate(0.5, 2.0, 300, 400)["heat_rate_w"] < 0
    with pytest.raises(ValueError):
        radiation_heat_rate(1.5, 1, 400, 300)


# ---- vibration --------------------------------------------------------------------------------

def test_damped_vibration_parameters():
    r = damped_vibration(1.0, 100.0, 1.0)
    assert r["natural_frequency_rad_s"] == pytest.approx(10.0)
    assert r["critical_damping_n_s_m"] == pytest.approx(20.0)
    assert r["damping_ratio"] == pytest.approx(0.05)
    assert r["damping_type"] == "underdamped"
    assert r["damped_frequency_hz"] == pytest.approx(10 * math.sqrt(1 - 0.05 ** 2) / (2 * math.pi))
    assert r["logarithmic_decrement"] == pytest.approx(2 * math.pi * 0.05 / math.sqrt(1 - 0.05 ** 2))
    assert damped_vibration(1, 100, 20)["damping_type"] == "critically damped"
    assert damped_vibration(1, 100, 40)["damping_type"] == "overdamped"
    assert damped_vibration(1, 100)["damping_type"] == "undamped"


def test_forced_vibration_and_isolation():
    wn_hz = 10 / (2 * math.pi)
    res = damped_vibration(1, 100, 1.0, forcing_frequency_hz=wn_hz)
    assert res["magnification_factor"] == pytest.approx(1 / (2 * 0.05))
    assert math.isinf(damped_vibration(1, 100, forcing_frequency_hz=wn_hz)["magnification_factor"])
    iso = damped_vibration(1, 100, forcing_frequency_hz=math.sqrt(2) * wn_hz)
    assert iso["transmissibility"] == pytest.approx(1.0)
    assert damped_vibration(1, 100, 1.0, forcing_frequency_hz=3 * wn_hz)["isolates_vibration"]
    assert not damped_vibration(1, 100, 1.0, forcing_frequency_hz=wn_hz)["isolates_vibration"]
    assert damped_vibration(1, 100, 1.0, forcing_frequency_hz=0)["magnification_factor"] == pytest.approx(1.0)


@pytest.mark.parametrize("c", [0.0, 1.0, 20.0, 40.0])
def test_free_response_satisfies_ode_and_initial_conditions(c):
    m, k, x0, v0, T = 2.0, 200.0, 0.01, 0.3, 2.0
    # c scaled for m = 2: critical c = 2*sqrt(k*m) = 40, so use c * 1 for under/critical and 80 for over
    c_val = 80.0 if c == 40.0 else (40.0 if c == 20.0 else c)
    r = damped_vibration(m, k, c_val, initial_displacement_m=x0, initial_velocity_m_s=v0, response_duration_s=T, samples=4001)
    t, x = r["time_s"], r["displacement_m"]
    assert x[0] == pytest.approx(x0)
    dt = t[1] - t[0]
    assert (x[1] - x[0]) / dt == pytest.approx(v0, rel=2e-2)
    for i in (500, 1500, 3000):
        v = (x[i + 1] - x[i - 1]) / (2 * dt)
        a = (x[i + 1] - 2 * x[i] + x[i - 1]) / dt ** 2
        assert m * a + c_val * v + k * x[i] == pytest.approx(0, abs=1e-2)


def test_vibration_errors():
    with pytest.raises(ValueError):
        damped_vibration(0, 100)
    with pytest.raises(ValueError):
        damped_vibration(1, 100, -1)
    with pytest.raises(ValueError):
        damped_vibration(1, 100, response_duration_s=0)
