import math
import pytest
from engineering_toolkit import (lewis_form_factor, spur_gear_geometry, spur_gear_stress,
                                 spur_gear_module_for_bending)


def test_geometry_basic_dimensions():
    g = spur_gear_geometry(0.002, 20, 40)
    assert g["pinion_pitch_diameter_m"] == pytest.approx(0.04)
    assert g["gear_pitch_diameter_m"] == pytest.approx(0.08)
    assert g["centre_distance_m"] == pytest.approx(0.06)
    assert g["pinion_outside_diameter_m"] == pytest.approx(0.044)
    assert g["gear_root_diameter_m"] == pytest.approx(0.075)
    assert g["pinion_base_diameter_m"] == pytest.approx(0.04 * math.cos(math.radians(20)))
    assert g["base_pitch_m"] == pytest.approx(math.pi * 0.002 * math.cos(math.radians(20)))
    assert g["speed_ratio"] == pytest.approx(2.0)


def test_contact_ratio_and_undercut_flag():
    assert spur_gear_geometry(0.002, 20, 40)["contact_ratio"] == pytest.approx(1.635, abs=0.01)
    assert 1.5 < spur_gear_geometry(0.003, 30, 30)["contact_ratio"] < 2.0
    assert spur_gear_geometry(0.002, 18, 40)["undercut_free"]
    assert not spur_gear_geometry(0.002, 17, 40)["undercut_free"]  # 17 < 2/sin^2(20 deg) = 17.1
    assert not spur_gear_geometry(0.002, 14, 40)["undercut_free"]
    assert spur_gear_geometry(0.002, 20, 40)["minimum_teeth_no_undercut"] == pytest.approx(17.1, abs=0.1)


def test_lewis_factor_table_and_interpolation():
    assert lewis_form_factor(20) == pytest.approx(0.322)
    assert lewis_form_factor(25) == pytest.approx((0.337 + 0.346) / 2)
    assert lewis_form_factor(1000) == pytest.approx(0.472)
    with pytest.raises(ValueError):
        lewis_form_factor(10)


def test_stress_forces_and_formulas():
    r = spur_gear_stress(5000, 1500, 0.002, 20, 40, 0.02, finish="hobbed")
    omega = 2 * math.pi * 1500 / 60
    assert r["pinion_torque_nm"] == pytest.approx(5000 / omega)
    assert r["tangential_force_n"] == pytest.approx(2 * 5000 / omega / 0.04)
    assert r["pitch_line_velocity_m_s"] == pytest.approx(omega * 0.02)
    assert r["velocity_factor"] == pytest.approx((6.1 + omega * 0.02) / 6.1)
    wt, kv = r["tangential_force_n"], r["velocity_factor"]
    assert r["bending_stress_pinion_pa"] == pytest.approx(wt * kv / (0.02 * 0.002 * 0.322))
    assert r["bending_stress_gear_pa"] < r["bending_stress_pinion_pa"]
    i_geo = math.sin(math.radians(20)) * math.cos(math.radians(20)) / 2 * (2 / 3)
    cp = math.sqrt(1 / (math.pi * 2 * (1 - 0.09) / 200e9))
    assert r["contact_stress_pa"] == pytest.approx(cp * math.sqrt(wt * kv / (0.02 * 0.04 * i_geo)))


def test_stress_scaling_and_safety_factors():
    base = spur_gear_stress(5000, 1500, 0.002, 20, 40, 0.02)
    wide = spur_gear_stress(5000, 1500, 0.002, 20, 40, 0.04)
    assert wide["bending_stress_pinion_pa"] == pytest.approx(base["bending_stress_pinion_pa"] / 2)
    assert wide["contact_stress_pa"] == pytest.approx(base["contact_stress_pa"] / math.sqrt(2))
    shock = spur_gear_stress(5000, 1500, 0.002, 20, 40, 0.02, application_factor=1.5)
    assert shock["bending_stress_pinion_pa"] == pytest.approx(1.5 * base["bending_stress_pinion_pa"])
    fos = spur_gear_stress(5000, 1500, 0.002, 20, 40, 0.02, allowable_bending_pa=200e6, allowable_contact_pa=600e6)
    assert fos["bending_factor_of_safety_pinion"] == pytest.approx(200e6 / base["bending_stress_pinion_pa"])
    assert fos["contact_factor_of_safety"] == pytest.approx(600e6 / base["contact_stress_pa"])
    assert spur_gear_stress(5000, 1500, 0.002, 20, 40, 0.02, finish="ground")["velocity_factor"] < base["velocity_factor"] * 1.2


def test_module_sizing_meets_allowable():
    r = spur_gear_module_for_bending(5000, 1500, 20, 100e6)
    check = spur_gear_stress(5000, 1500, r["module_m"], 20, 40, r["face_width_m"])
    assert check["bending_stress_pinion_pa"] == pytest.approx(100e6, rel=1e-6)
    assert r["recommended_standard_module_m"] >= r["module_m"]
    assert r["recommended_standard_module_m"] == pytest.approx(0.003)
    stronger = spur_gear_module_for_bending(5000, 1500, 20, 200e6)
    assert stronger["module_m"] < r["module_m"]


def test_gear_errors():
    with pytest.raises(ValueError):
        spur_gear_geometry(0.002, 20.5, 40)
    with pytest.raises(ValueError):
        spur_gear_geometry(0.002, 20, 40, pressure_angle_deg=45)
    with pytest.raises(ValueError):
        spur_gear_stress(5000, 1500, 0.002, 20, 40, 0.02, finish="rough")
    with pytest.raises(ValueError):
        spur_gear_stress(-1, 1500, 0.002, 20, 40, 0.02)
    with pytest.raises(ValueError):
        spur_gear_module_for_bending(5000, 1500, 20, 100e6, face_width_to_module=0)
