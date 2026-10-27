import math
import pytest
import engineering_toolkit as et

def test_compression_spring_outputs_positive_rate_and_stress():
    r = et.compression_spring_design(500, .005, .04, 8, 79e9, free_length_m=.1)
    assert r["spring_rate_n_per_m"] > 0
    assert r["maximum_shear_stress_pa"] > 0
    assert r["estimated_solid_clearance_m"] > 0

def test_belt_drive_power_and_speed_ratio():
    r = et.belt_drive_design(1500, .2, .4, 1440, .3, math.radians(170))
    assert r["driven_speed_rpm_ideal"] == pytest.approx(720)
    assert r["effective_tension_n"] * r["belt_speed_m_s"] == pytest.approx(1500)

def test_power_screw_self_locking_and_torque():
    r = et.power_screw_design(10000, .03, .005, .15)
    assert r["torque_to_raise_nm"] > 0
    assert r["self_locking_predicted"] is True

def test_fatigue_shaft_diameter_positive_and_safe_check_larger():
    r = et.shaft_fatigue_design(250, 100, 30, 200e6, 600e6, 400e6)
    assert r["required_diameter_mm"] > 0
    assert r["required_diameter_mm"] >= r["yield_check_diameter_mm"]

def test_invalid_spring_geometry_rejected():
    with pytest.raises(ValueError):
        et.compression_spring_design(500, .01, .005, 8, 79e9)
