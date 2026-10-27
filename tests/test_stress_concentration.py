import math
import pytest
from engineering_toolkit import (fatigue_notch_factor, goodman_fatigue, notch_sensitivity,
                                 stress_concentration_factor)


def test_small_hole_in_wide_plate_approaches_three():
    r = stress_concentration_factor("circular_hole_finite_plate", width_m=1.0, hole_diameter_m=1e-6)
    assert r["kt"] == pytest.approx(3.0, abs=1e-4)


def test_finite_plate_hole_matches_polynomial_by_hand():
    x = 0.5
    expected = 3.00 - 3.13 * x + 3.66 * x ** 2 - 1.53 * x ** 3
    r = stress_concentration_factor("circular_hole_finite_plate", width_m=0.10, hole_diameter_m=0.05)
    assert r["kt"] == pytest.approx(expected) and r["kt"] == pytest.approx(2.15875)
    assert r["diameter_to_width_ratio"] == pytest.approx(0.5)


def test_kt_falls_as_the_hole_gets_relatively_larger():
    kts = [stress_concentration_factor("circular_hole_finite_plate", width_m=1.0, hole_diameter_m=d)["kt"]
           for d in (0.05, 0.2, 0.4, 0.6)]
    assert kts == sorted(kts, reverse=True)


def test_elliptical_hole_inglis_solution():
    assert stress_concentration_factor("elliptical_hole_infinite_plate", semi_axis_transverse_m=0.01,
                                       semi_axis_parallel_m=0.01)["kt"] == pytest.approx(3.0)
    assert stress_concentration_factor("elliptical_hole_infinite_plate", semi_axis_transverse_m=0.02,
                                       semi_axis_parallel_m=0.01)["kt"] == pytest.approx(5.0)
    # an ellipse lying along the load (a < b) is milder than a circle
    assert stress_concentration_factor("elliptical_hole_infinite_plate", semi_axis_transverse_m=0.005,
                                       semi_axis_parallel_m=0.01)["kt"] == pytest.approx(2.0)


def test_peak_stress_is_kt_times_nominal():
    r = stress_concentration_factor("circular_hole_finite_plate", width_m=0.1, hole_diameter_m=0.02,
                                    nominal_stress_pa=120e6)
    assert r["peak_stress_pa"] == pytest.approx(r["kt"] * 120e6)
    assert "peak_stress_pa" not in stress_concentration_factor(
        "circular_hole_finite_plate", width_m=0.1, hole_diameter_m=0.02)


def test_neuber_notch_sensitivity_matches_hand_calculation():
    sut_kpsi, r_in = 100.0, 0.1
    sqrt_a = 0.246 - 3.08e-3 * sut_kpsi + 1.51e-5 * sut_kpsi ** 2 - 2.67e-8 * sut_kpsi ** 3
    expected = 1 / (1 + sqrt_a / math.sqrt(r_in))
    got = notch_sensitivity(r_in * 0.0254, sut_kpsi * 6.894757293168e6)
    assert got["q"] == pytest.approx(expected)
    assert 0.75 < got["q"] < 0.9          # typical value for a 100 kpsi steel with a 0.1 in radius


def test_torsion_uses_its_own_smaller_material_constant():
    args = (0.002, 800e6)
    bend, tors = notch_sensitivity(*args, loading="bending"), notch_sensitivity(*args, loading="torsion")
    assert tors["material_constant_sqrt_a_sqrt_in"] < bend["material_constant_sqrt_a_sqrt_in"]
    assert tors["q"] > bend["q"]
    assert notch_sensitivity(*args, loading="axial")["q"] == pytest.approx(bend["q"])


def test_peterson_matches_hand_calculation():
    a_mm = 0.0254 * (2070 / 700) ** 1.8
    got = notch_sensitivity(0.002, 700e6, method="peterson")
    assert got["q"] == pytest.approx(1 / (1 + a_mm / 2.0))


def test_larger_radius_and_stronger_steel_raise_q():
    assert notch_sensitivity(0.005, 800e6)["q"] > notch_sensitivity(0.001, 800e6)["q"]
    assert notch_sensitivity(0.002, 1200e6, method="peterson")["q"] > notch_sensitivity(0.002, 600e6, method="peterson")["q"]


def test_kf_follows_its_definition_and_lies_between_one_and_kt():
    q = notch_sensitivity(0.003, 900e6)["q"]
    r = fatigue_notch_factor(2.5, 0.003, 900e6)
    assert r["kf"] == pytest.approx(1 + q * 1.5)
    assert 1.0 < r["kf"] < 2.5
    assert fatigue_notch_factor(1.0, 0.003, 900e6)["kf"] == pytest.approx(1.0)


def test_kf_can_feed_the_goodman_check():
    kf = fatigue_notch_factor(2.0, 0.002, 700e6)["kf"]
    plain = goodman_fatigue(700e6, 250e6, stress_mean_pa=0.0, stress_amplitude_pa=100e6)["goodman_factor_of_safety"]
    notched = goodman_fatigue(700e6, 250e6, stress_mean_pa=0.0, stress_amplitude_pa=kf * 100e6)["goodman_factor_of_safety"]
    assert notched == pytest.approx(plain / kf)


@pytest.mark.parametrize("call", [
    lambda: stress_concentration_factor("square_hole"),
    lambda: stress_concentration_factor("circular_hole_finite_plate", width_m=0.1),
    lambda: stress_concentration_factor("circular_hole_finite_plate", width_m=0.1, hole_diameter_m=0.08),
    lambda: stress_concentration_factor("circular_hole_finite_plate", width_m=-0.1, hole_diameter_m=0.01),
    lambda: stress_concentration_factor("elliptical_hole_infinite_plate", semi_axis_transverse_m=0.01),
    lambda: stress_concentration_factor("elliptical_hole_infinite_plate", semi_axis_transverse_m=0.01, semi_axis_parallel_m=0),
    lambda: stress_concentration_factor("circular_hole_finite_plate", width_m=0.1, hole_diameter_m=0.01, nominal_stress_pa=float("nan")),
    lambda: notch_sensitivity(0, 800e6),
    lambda: notch_sensitivity(0.002, 100e6),
    lambda: notch_sensitivity(0.002, 2000e6),
    lambda: notch_sensitivity(0.002, 800e6, loading="shear"),
    lambda: notch_sensitivity(0.002, 800e6, method="made_up"),
    lambda: notch_sensitivity(0.002, 800e6, loading="torsion", method="peterson"),
    lambda: fatigue_notch_factor(0.9, 0.002, 800e6),
    lambda: fatigue_notch_factor(float("inf"), 0.002, 800e6),
])
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
