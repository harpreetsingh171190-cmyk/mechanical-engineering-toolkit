import math
import pytest
from engineering_toolkit import (
    rectangular_section_properties, beam_deflection_point_load, euler_buckling_load,
    von_mises_stress, torsion_solid_shaft, shaft_diameter_for_torque,
    thin_wall_pressure_vessel, carnot_efficiency, simply_supported_beam_point_load,
)


def test_rectangular_section():
    r = rectangular_section_properties(0.1, 0.2)
    assert r["area_m2"] == pytest.approx(0.02)
    assert r["second_moment_m4"] == pytest.approx(0.1 * 0.2**3 / 12)
    assert r["section_modulus_m3"] == pytest.approx(0.1 * 0.2**2 / 6)


def test_midspan_deflection_matches_textbook():
    P, L, E, I = 1000.0, 4.0, 200e9, 8e-6
    r = beam_deflection_point_load(L, P, L / 2, E, I)
    expected = P * L**3 / (48 * E * I)
    assert r["maximum_deflection_m"] == pytest.approx(expected)
    assert r["midspan_deflection_m"] == pytest.approx(expected)
    assert max(r["deflection_m"]) == pytest.approx(expected, rel=1e-6)
    assert r["maximum_deflection_location_m"] == pytest.approx(L / 2)


@pytest.mark.parametrize("a", [0.5, 1.0, 3.0, 3.5])
def test_off_center_deflection_max_matches_sampling(a):
    r = beam_deflection_point_load(4.0, 1000.0, a, 200e9, 8e-6, samples=40001)
    assert max(r["deflection_m"]) == pytest.approx(r["maximum_deflection_m"], rel=1e-6)
    i = r["deflection_m"].index(max(r["deflection_m"]))
    assert r["x_m"][i] == pytest.approx(r["maximum_deflection_location_m"], abs=1e-3)


def test_deflection_zero_at_supports():
    r = beam_deflection_point_load(3.0, 500.0, 1.0, 70e9, 1e-6)
    assert r["deflection_m"][0] == pytest.approx(0, abs=1e-15)
    assert r["deflection_m"][-1] == pytest.approx(0, abs=1e-15)


def test_euler_buckling():
    r = euler_buckling_load(200e9, 1e-6, 2.0)
    assert r["critical_load_n"] == pytest.approx(math.pi**2 * 200e9 * 1e-6 / 4)
    fixed = euler_buckling_load(200e9, 1e-6, 2.0, 0.5)
    assert fixed["critical_load_n"] == pytest.approx(4 * r["critical_load_n"])


def test_von_mises_cases():
    assert von_mises_stress(100, 0, 0) == pytest.approx(100)
    assert von_mises_stress(0, 0, 50) == pytest.approx(50 * math.sqrt(3))
    assert von_mises_stress(80, 80, 0) == pytest.approx(80)


def test_torsion_and_sizing_round_trip():
    d = shaft_diameter_for_torque(500, 60e6)
    assert torsion_solid_shaft(500, d)["max_shear_stress_pa"] == pytest.approx(60e6)


def test_pressure_vessel():
    r = thin_wall_pressure_vessel(1e6, 0.5, 0.01)
    assert r["cylinder_hoop_stress_pa"] == pytest.approx(50e6)
    assert r["cylinder_longitudinal_stress_pa"] == pytest.approx(25e6)
    assert r["sphere_stress_pa"] == pytest.approx(25e6)
    assert r["thin_wall_valid"] is True
    assert thin_wall_pressure_vessel(1e6, 0.05, 0.01)["thin_wall_valid"] is False


def test_carnot():
    assert carnot_efficiency(600, 300) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        carnot_efficiency(300, 300)


def test_load_at_support_gives_no_span_shear():
    for a in (0.0, 4.0):
        r = simply_supported_beam_point_load(4, 1000, a, samples=9)
        assert all(abs(v) < 1e-9 for v in r["shear_n"])
        assert max(abs(m) for m in r["moment_nm"]) < 1e-9


def test_new_inputs_rejected():
    with pytest.raises(ValueError):
        euler_buckling_load(200e9, 0, 2)
    with pytest.raises(ValueError):
        thin_wall_pressure_vessel(1e6, 0.5, 0)
    with pytest.raises(ValueError):
        beam_deflection_point_load(4, 1000, 5, 200e9, 8e-6)
