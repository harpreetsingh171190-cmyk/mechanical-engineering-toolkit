import math
import pytest
from engineering_toolkit import (
    thermal_expansion, shaft_torque, spring_mass_frequency,
    factor_of_safety, simply_supported_beam_point_load, convert_units,
)


def test_thermal_expansion():
    r = thermal_expansion(2.0, 12e-6, 100)
    assert r["delta_length_m"] == pytest.approx(0.0024)
    assert r["final_length_m"] == pytest.approx(2.0024)


def test_shaft_torque():
    r = shaft_torque(1000, 60)
    assert r["angular_speed_rad_s"] == pytest.approx(2 * math.pi)
    assert r["torque_nm"] == pytest.approx(1000 / (2 * math.pi))


def test_spring_frequency():
    r = spring_mass_frequency(100, 1)
    assert r["frequency_hz"] == pytest.approx(10 / (2 * math.pi))
    assert r["period_s"] == pytest.approx(2 * math.pi / 10)


def test_factor_of_safety():
    assert factor_of_safety(300e6, 100e6) == pytest.approx(3.0)


def test_beam_reactions_and_max_moment():
    r = simply_supported_beam_point_load(4, 1000, 2)
    assert r["reaction_left_n"] == pytest.approx(500)
    assert r["reaction_right_n"] == pytest.approx(500)
    assert r["maximum_moment_nm"] == pytest.approx(1000)
    assert max(r["moment_nm"]) == pytest.approx(1000)


def test_unit_conversion():
    assert convert_units(1, "m", "mm") == pytest.approx(1000)
    assert convert_units(0, "C", "F") == pytest.approx(32)
    assert convert_units(1, "bar", "kPa") == pytest.approx(100)


@pytest.mark.parametrize("args", [(0, 1, 0), (1, -1, 0), (1, 1, 2)])
def test_bad_beam_inputs(args):
    with pytest.raises(ValueError):
        simply_supported_beam_point_load(*args)


def test_invalid_units():
    with pytest.raises(ValueError):
        convert_units(1, "m", "MPa")


def test_reynolds_number():
    from engineering_toolkit import reynolds_number
    assert reynolds_number(1000, 2, 0.05, 0.001) == pytest.approx(100000)

def test_pump_power():
    from engineering_toolkit import pump_power
    r = pump_power(0.01, 100000, 0.8)
    assert r["hydraulic_power_w"] == pytest.approx(1000)
    assert r["input_power_w"] == pytest.approx(1250)

def test_conduction_heat_rate():
    from engineering_toolkit import conduction_heat_rate
    assert conduction_heat_rate(10, 2, 100, 20, 0.5) == pytest.approx(3200)

def test_gear_train():
    from engineering_toolkit import gear_train_output
    r = gear_train_output(1200, 20, 60, 10, 0.9)
    assert r["output_rpm"] == pytest.approx(400)
    assert r["output_torque_nm"] == pytest.approx(27)

def test_heat_engine_performance():
    from engineering_toolkit import heat_engine_performance
    r = heat_engine_performance(1000, 600)
    assert r["net_power_w"] == pytest.approx(400)
    assert r["thermal_efficiency_percent"] == pytest.approx(40)

def test_invalid_pump_efficiency():
    from engineering_toolkit import pump_power
    with pytest.raises(ValueError):
        pump_power(0.1, 1000, 1.2)
