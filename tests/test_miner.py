import math
import pytest
from engineering_toolkit import miner_cumulative_damage, sn_fatigue_life

SUT, SE = 600e6, 200e6


def test_single_block_damage_is_cycles_over_life():
    life = sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=300e6)["life_cycles"]
    r = miner_cumulative_damage(SUT, SE, [(0.0, 300e6, life / 4)])
    assert r["total_damage"] == pytest.approx(0.25) and r["blocks_to_failure"] == pytest.approx(4.0)
    assert r["failure_predicted"] is False


def test_damage_adds_across_blocks_and_hits_one_at_failure():
    l1 = sn_fatigue_life(SUT, SE, stress_mean_pa=0.0, stress_amplitude_pa=300e6)["life_cycles"]
    l2 = sn_fatigue_life(SUT, SE, stress_mean_pa=50e6, stress_amplitude_pa=250e6)["life_cycles"]
    r = miner_cumulative_damage(SUT, SE, [(0.0, 300e6, 0.5 * l1), (50e6, 250e6, 0.5 * l2)])
    assert r["total_damage"] == pytest.approx(1.0) and r["failure_predicted"] is True


def test_blocks_below_endurance_limit_add_no_damage():
    r = miner_cumulative_damage(SUT, SE, [(0.0, 100e6, 1e9)])
    assert r["total_damage"] == 0.0 and math.isinf(r["blocks_to_failure"])


def test_notch_factor_increases_damage():
    a = miner_cumulative_damage(SUT, SE, [(0.0, 150e6, 1e5)], notch_factor_kf=1.0)["total_damage"]
    b = miner_cumulative_damage(SUT, SE, [(0.0, 150e6, 1e5)], notch_factor_kf=2.0)["total_damage"]
    assert a == 0.0 and b > 0.0


@pytest.mark.parametrize("call", [
    lambda: miner_cumulative_damage(SUT, SE, []),
    lambda: miner_cumulative_damage(SUT, SE, [(0.0, 1e6)]),
    lambda: miner_cumulative_damage(SUT, SE, [(0.0, 1e6, 0)]),
    lambda: miner_cumulative_damage(SUT, SE, [(0.0, -1e6, 10)]),
])
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
