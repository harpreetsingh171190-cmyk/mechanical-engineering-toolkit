import math
import pytest
from engineering_toolkit import sn_curve, sn_fatigue_life

SUT, SE = 600e6, 200e6
A = (0.9 * SUT) ** 2 / SE
B = -math.log10(0.9 * SUT / SE) / 3


def test_curve_passes_through_both_anchor_points():
    assert A * 1e3 ** B == pytest.approx(0.9 * SUT)
    assert A * 1e6 ** B == pytest.approx(SE)
    r = sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=0.9 * SUT)
    assert r["life_cycles"] == pytest.approx(1e3)
    assert r["sn_coefficient_a_pa"] == pytest.approx(A) and r["sn_exponent_b"] == pytest.approx(B)


def test_fully_reversed_life_matches_hand_calculation():
    r = sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=300e6)
    assert r["life_cycles"] == pytest.approx((300e6 / A) ** (1 / B))
    assert 5e4 < r["life_cycles"] < 8e4
    assert r["infinite_life_predicted"] is False and r["high_cycle_range_valid"] is True


def test_stress_at_or_below_endurance_limit_gives_infinite_life():
    for amp in (SE, 0.5 * SE, 0.0):
        r = sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=amp)
        assert math.isinf(r["life_cycles"]) and r["infinite_life_predicted"] is True


def test_tensile_mean_stress_shortens_life_via_goodman():
    rev = sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=250e6)["life_cycles"]
    with_mean = sn_fatigue_life(SUT, SE, stress_mean_pa=100e6, stress_amplitude_pa=250e6)
    assert with_mean["equivalent_reversed_stress_pa"] == pytest.approx(250e6 / (1 - 100 / 600))
    assert with_mean["life_cycles"] < rev


def test_compressive_mean_stress_is_ignored():
    a = sn_fatigue_life(SUT, SE, stress_mean_pa=-150e6, stress_amplitude_pa=280e6)
    b = sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=280e6)
    assert a["life_cycles"] == pytest.approx(b["life_cycles"])


def test_max_min_and_mean_amplitude_forms_agree():
    a = sn_fatigue_life(SUT, SE, stress_max_pa=350e6, stress_min_pa=-50e6)
    b = sn_fatigue_life(SUT, SE, stress_mean_pa=150e6, stress_amplitude_pa=200e6)
    assert a["life_cycles"] == pytest.approx(b["life_cycles"])


def test_notch_factor_multiplies_the_stress_and_shortens_life():
    plain = sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=150e6)
    notched = sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=150e6, notch_factor_kf=2.0)
    assert plain["infinite_life_predicted"] is True
    assert notched["equivalent_reversed_stress_pa"] == pytest.approx(300e6)
    assert notched["life_cycles"] == pytest.approx((300e6 / A) ** (1 / B))
    assert notched["stress_amplitude_pa"] == pytest.approx(150e6)      # nominal stress is reported unchanged


def test_life_below_1000_cycles_is_flagged():
    r = sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=0.95 * SUT)
    assert r["life_cycles"] < 1e3 and r["high_cycle_range_valid"] is False


def test_target_life_factor_of_safety_and_allowable_amplitude():
    r = sn_fatigue_life(SUT, SE, stress_mean_pa=100e6, stress_amplitude_pa=200e6, target_cycles=1e5)
    strength = A * 1e5 ** B
    assert r["fatigue_strength_at_target_pa"] == pytest.approx(strength)
    assert r["factor_of_safety_at_target"] == pytest.approx(strength / r["equivalent_reversed_stress_pa"])
    # running at the allowable amplitude should give life exactly equal to the target
    again = sn_fatigue_life(SUT, SE, stress_mean_pa=100e6, stress_amplitude_pa=r["allowable_amplitude_at_target_pa"])
    assert again["life_cycles"] == pytest.approx(1e5, rel=1e-9)


def test_target_beyond_a_million_cycles_uses_the_endurance_limit():
    r = sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=100e6, target_cycles=1e8)
    assert r["fatigue_strength_at_target_pa"] == pytest.approx(SE)
    assert r["factor_of_safety_at_target"] == pytest.approx(2.0)


def test_custom_fatigue_strength_fraction_changes_the_slope():
    r = sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=300e6, fatigue_strength_fraction=0.8)
    a = (0.8 * SUT) ** 2 / SE
    b = -math.log10(0.8 * SUT / SE) / 3
    assert r["life_cycles"] == pytest.approx((300e6 / a) ** (1 / b))


def test_sn_curve_shape():
    c = sn_curve(SUT, SE, points=41, max_cycles=1e7)
    assert c["cycles"][0] == pytest.approx(1e3) and c["cycles"][-1] == pytest.approx(1e7)
    assert c["strength_pa"][0] == pytest.approx(0.9 * SUT)
    assert all(x >= y for x, y in zip(c["strength_pa"], c["strength_pa"][1:]))
    assert c["strength_pa"][-1] == pytest.approx(SE)
    assert all(s == pytest.approx(SE) for n, s in zip(c["cycles"], c["strength_pa"]) if n >= 1e6)


@pytest.mark.parametrize("call", [
    lambda: sn_fatigue_life(SUT, SE),
    lambda: sn_fatigue_life(SUT, SE, stress_mean_pa=0.0),
    lambda: sn_fatigue_life(0, SE, stress_mean_pa=0.0, stress_amplitude_pa=1e6),
    lambda: sn_fatigue_life(SUT, SUT * 1.2, stress_mean_pa=0.0, stress_amplitude_pa=1e6),
    lambda: sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=-1e6),
    lambda: sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=1e6, notch_factor_kf=0.8),
    lambda: sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=1e6, fatigue_strength_fraction=0),
    lambda: sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=1e6, fatigue_strength_fraction=1.2),
    lambda: sn_fatigue_life(SUT, 0.95 * SUT, stress_mean_pa=0.0, stress_amplitude_pa=1e6),   # f*Sut below Se
    lambda: sn_fatigue_life(SUT, SE, stress_mean_pa=SUT, stress_amplitude_pa=1e6),
    lambda: sn_fatigue_life(SUT, SE, stress_mean_pa=300e6, stress_amplitude_pa=1e6, notch_factor_kf=2.0),
    lambda: sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=1e6, target_cycles=500),
    lambda: sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=1e6, target_cycles=float("nan")),
    lambda: sn_curve(SUT, SE, points=1),
    lambda: sn_curve(SUT, SE, max_cycles=500),
    lambda: sn_curve(SUT, 0.99 * SUT),
])
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
