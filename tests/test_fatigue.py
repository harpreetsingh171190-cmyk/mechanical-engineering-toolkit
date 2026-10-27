import math
import pytest
from engineering_toolkit import goodman_fatigue

SUT, SE, SY = 600e6, 200e6, 400e6


def test_goodman_matches_hand_calculation():
    r = goodman_fatigue(SUT, SE, stress_mean_pa=100e6, stress_amplitude_pa=100e6)
    assert r["goodman_factor_of_safety"] == pytest.approx(1 / (100 / 200 + 100 / 600))
    assert r["infinite_life_predicted"] is True


def test_max_min_input_gives_mean_and_amplitude():
    r = goodman_fatigue(SUT, SE, stress_max_pa=300e6, stress_min_pa=-100e6, yield_strength_pa=SY)
    assert r["stress_mean_pa"] == pytest.approx(100e6) and r["stress_amplitude_pa"] == pytest.approx(200e6)
    assert r["stress_ratio"] == pytest.approx(-1 / 3)
    assert r["goodman_factor_of_safety"] == pytest.approx(1 / (200 / 200 + 100 / 600))
    assert r["infinite_life_predicted"] is False
    assert r["yield_factor_of_safety"] == pytest.approx(SY / 300e6)


def test_both_input_forms_agree():
    a = goodman_fatigue(SUT, SE, stress_max_pa=250e6, stress_min_pa=50e6)
    b = goodman_fatigue(SUT, SE, stress_mean_pa=150e6, stress_amplitude_pa=100e6)
    assert a["goodman_factor_of_safety"] == pytest.approx(b["goodman_factor_of_safety"])


def test_fully_reversed_loading_is_endurance_limit_over_amplitude():
    r = goodman_fatigue(SUT, SE, stress_max_pa=150e6, stress_min_pa=-150e6)
    assert r["goodman_factor_of_safety"] == pytest.approx(SE / 150e6)
    assert r["stress_ratio"] == pytest.approx(-1.0)


def test_point_on_the_goodman_line_has_factor_one():
    sm = 300e6
    sa = SE * (1 - sm / SUT)
    r = goodman_fatigue(SUT, SE, stress_mean_pa=sm, stress_amplitude_pa=sa)
    assert r["goodman_factor_of_safety"] == pytest.approx(1.0)
    assert r["allowable_amplitude_at_mean_pa"] == pytest.approx(sa)


def test_compressive_mean_stress_uses_the_flat_line():
    r = goodman_fatigue(SUT, SE, stress_mean_pa=-100e6, stress_amplitude_pa=100e6)
    assert r["goodman_factor_of_safety"] == pytest.approx(2.0)
    assert r["allowable_amplitude_at_mean_pa"] == pytest.approx(SE)


def test_static_load_has_infinite_fatigue_factor():
    r = goodman_fatigue(SUT, SE, stress_max_pa=100e6, stress_min_pa=100e6, yield_strength_pa=SY)
    assert math.isinf(r["goodman_factor_of_safety"]) and r["infinite_life_predicted"] is True
    assert r["yield_factor_of_safety"] == pytest.approx(4.0)


def test_zero_stress_cycle_does_not_divide_by_zero():
    r = goodman_fatigue(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=0.0, yield_strength_pa=SY)
    assert math.isinf(r["goodman_factor_of_safety"]) and math.isinf(r["yield_factor_of_safety"])
    assert r["stress_ratio"] is None


def test_higher_mean_stress_lowers_the_factor():
    low = goodman_fatigue(SUT, SE, stress_mean_pa=50e6, stress_amplitude_pa=80e6)["goodman_factor_of_safety"]
    high = goodman_fatigue(SUT, SE, stress_mean_pa=250e6, stress_amplitude_pa=80e6)["goodman_factor_of_safety"]
    assert high < low


@pytest.mark.parametrize("call", [
    lambda: goodman_fatigue(0, SE, stress_mean_pa=1, stress_amplitude_pa=1),
    lambda: goodman_fatigue(SUT, -1, stress_mean_pa=1, stress_amplitude_pa=1),
    lambda: goodman_fatigue(SUT, SUT * 1.1, stress_mean_pa=1, stress_amplitude_pa=1),
    lambda: goodman_fatigue(SUT, SE),
    lambda: goodman_fatigue(SUT, SE, stress_max_pa=1e6),
    lambda: goodman_fatigue(SUT, SE, stress_mean_pa=1e6),
    lambda: goodman_fatigue(SUT, SE, stress_max_pa=1e6, stress_min_pa=0, stress_mean_pa=1e6, stress_amplitude_pa=1e6),
    lambda: goodman_fatigue(SUT, SE, stress_max_pa=-5e6, stress_min_pa=5e6),
    lambda: goodman_fatigue(SUT, SE, stress_mean_pa=1e6, stress_amplitude_pa=-1),
    lambda: goodman_fatigue(SUT, SE, stress_max_pa=float("nan"), stress_min_pa=0),
    lambda: goodman_fatigue(SUT, SE, stress_mean_pa=1e6, stress_amplitude_pa=1e6, yield_strength_pa=0),
])
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
