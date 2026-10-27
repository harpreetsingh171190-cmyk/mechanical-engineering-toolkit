import math
import pytest
from engineering_toolkit import (log_mean_temperature_difference, heat_exchanger_lmtd,
                                 shell_and_tube_correction_factor, exchanger_effectiveness,
                                 exchanger_ntu, heat_exchanger_ntu, NTU_ARRANGEMENTS)


# ---- LMTD -------------------------------------------------------------------------------------

def test_lmtd_formula_and_equal_differences():
    assert log_mean_temperature_difference(50, 40) == pytest.approx(10 / math.log(50 / 40))
    assert log_mean_temperature_difference(30, 30) == pytest.approx(30)
    assert log_mean_temperature_difference(30, 30 + 1e-12) == pytest.approx(30)
    assert log_mean_temperature_difference(50, 40) == pytest.approx(log_mean_temperature_difference(40, 50))


def test_counterflow_versus_parallel_flow():
    cf = heat_exchanger_lmtd(100, 60, 20, 50, arrangement="counterflow")
    pf = heat_exchanger_lmtd(100, 60, 20, 50, arrangement="parallel_flow")
    assert (cf["delta_t1_k"], cf["delta_t2_k"]) == (50, 40)
    assert (pf["delta_t1_k"], pf["delta_t2_k"]) == (80, 10)
    assert cf["lmtd_k"] > pf["lmtd_k"]
    assert pf["lmtd_k"] == pytest.approx(70 / math.log(8))


def test_duty_area_and_u_round_trip():
    area = heat_exchanger_lmtd(100, 60, 20, 50, overall_u_w_m2k=500, heat_duty_w=50e3)["required_area_m2"]
    assert area == pytest.approx(50e3 / (500 * 44.8142), rel=1e-5)
    duty = heat_exchanger_lmtd(100, 60, 20, 50, overall_u_w_m2k=500, area_m2=area)["heat_duty_w"]
    assert duty == pytest.approx(50e3)
    u = heat_exchanger_lmtd(100, 60, 20, 50, area_m2=area, heat_duty_w=50e3)["required_overall_u_w_m2k"]
    assert u == pytest.approx(500)


def test_correction_factor_increases_required_area():
    base = heat_exchanger_lmtd(150, 90, 30, 80, overall_u_w_m2k=400, heat_duty_w=100e3)["required_area_m2"]
    f = shell_and_tube_correction_factor(150, 90, 30, 80)
    corrected = heat_exchanger_lmtd(150, 90, 30, 80, correction_factor=f, overall_u_w_m2k=400, heat_duty_w=100e3)["required_area_m2"]
    assert 0 < f < 1 and corrected == pytest.approx(base / f)


def test_shell_and_tube_factor_known_value_and_continuity_at_R_equals_1():
    assert shell_and_tube_correction_factor(150, 90, 30, 80) == pytest.approx(0.86693, abs=1e-4)
    at_one = shell_and_tube_correction_factor(150, 100, 30, 80)
    near_one = shell_and_tube_correction_factor(150, 100.000001, 30, 80)
    assert at_one == pytest.approx(near_one, abs=1e-5)


def test_correction_factor_approaches_one_for_small_temperature_change_on_one_side():
    # a nearly isothermal hot stream (condensing) behaves like pure counterflow: F -> 1
    assert shell_and_tube_correction_factor(150, 149.99, 30, 80) == pytest.approx(1.0, abs=1e-3)


def test_temperature_cross_and_bad_streams_are_rejected():
    for call in (
        lambda: heat_exchanger_lmtd(100, 60, 20, 100),                      # cold outlet = hot inlet: dT1 = 0
        lambda: heat_exchanger_lmtd(100, 60, 20, 110),                      # temperature cross
        lambda: heat_exchanger_lmtd(100, 120, 20, 50),                      # hot stream heats up
        lambda: heat_exchanger_lmtd(100, 60, 50, 20),                       # cold stream cools
        lambda: heat_exchanger_lmtd(100, 60, 20, 50, correction_factor=1.2),
        lambda: heat_exchanger_lmtd(100, 60, 20, 50, correction_factor=0),
        lambda: heat_exchanger_lmtd(100, 60, 20, 50, arrangement="crossflow"),
        lambda: heat_exchanger_lmtd(100, 60, 20, 50, overall_u_w_m2k=1, area_m2=1, heat_duty_w=1),
        lambda: heat_exchanger_lmtd(100, 60, 20, float("nan")),
        lambda: shell_and_tube_correction_factor(100, 60, 20, 110),
        lambda: shell_and_tube_correction_factor(100, 120, 20, 50),
        lambda: log_mean_temperature_difference(0, 10),
    ):
        with pytest.raises(ValueError):
            call()


# ---- effectiveness-NTU ------------------------------------------------------------------------

def test_known_effectiveness_values():
    assert exchanger_effectiveness(2.0, 1.0, "counterflow") == pytest.approx(2 / 3)
    assert exchanger_effectiveness(2.0, 1.0, "parallel_flow") == pytest.approx((1 - math.exp(-4)) / 2)
    assert exchanger_effectiveness(1.5, 0.5, "counterflow") == pytest.approx(
        (1 - math.exp(-0.75)) / (1 - 0.5 * math.exp(-0.75)))
    assert exchanger_effectiveness(2.0, 1.0, "shell_and_tube") == pytest.approx(0.55681, abs=1e-5)


@pytest.mark.parametrize("arrangement", NTU_ARRANGEMENTS)
def test_boiling_or_condensing_limit_is_arrangement_independent(arrangement):
    assert exchanger_effectiveness(1.7, 0.0, arrangement) == pytest.approx(1 - math.exp(-1.7))


@pytest.mark.parametrize("arrangement", NTU_ARRANGEMENTS)
@pytest.mark.parametrize("cr", [0.0, 0.25, 0.7, 1.0])
def test_effectiveness_is_monotonic_bounded_and_invertible(arrangement, cr):
    previous = 0.0
    for ntu in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0):
        eps = exchanger_effectiveness(ntu, cr, arrangement)
        assert previous < eps <= 1.0 and eps <= 1 / (1 + cr) + 1e-12 if arrangement == "parallel_flow" else previous < eps < 1.0
        assert exchanger_ntu(eps, cr, arrangement) == pytest.approx(ntu, rel=1e-6)
        previous = eps
    assert exchanger_effectiveness(0.0, cr, arrangement) == 0.0


def test_counterflow_beats_other_arrangements_and_parallel_is_worst():
    values = {a: exchanger_effectiveness(2.0, 0.6, a) for a in NTU_ARRANGEMENTS}
    assert max(values, key=values.get) == "counterflow"
    assert min(values, key=values.get) == "parallel_flow"


def test_counterflow_ntu_closed_form():
    eps, cr = 0.7, 0.5
    expected = math.log((1 - eps * cr) / (1 - eps)) / (1 - cr)
    assert exchanger_ntu(eps, cr, "counterflow") == pytest.approx(expected, rel=1e-9)


def test_parallel_flow_has_an_effectiveness_ceiling():
    with pytest.raises(ValueError):
        exchanger_ntu(0.6, 1.0, "parallel_flow")      # ceiling is 0.5
    assert exchanger_ntu(0.49, 1.0, "parallel_flow") > 0


def test_rating_problem_energy_balance_and_outlets():
    r = heat_exchanger_ntu(2000, 4000, 100, 20, ua_w_k=3000)
    assert r["c_min_w_k"] == 2000 and r["capacity_ratio"] == pytest.approx(0.5) and r["ntu"] == pytest.approx(1.5)
    assert r["heat_rate_w"] == pytest.approx(r["effectiveness"] * 2000 * 80)
    assert r["hot_outlet_c"] == pytest.approx(100 - r["heat_rate_w"] / 2000)
    assert r["cold_outlet_c"] == pytest.approx(20 + r["heat_rate_w"] / 4000)


def test_sizing_problem_round_trips_with_rating_problem():
    sized = heat_exchanger_ntu(3000, 1500, 90, 25, effectiveness=0.75, arrangement="shell_and_tube")
    rated = heat_exchanger_ntu(3000, 1500, 90, 25, ua_w_k=sized["ua_w_k"], arrangement="shell_and_tube")
    assert rated["effectiveness"] == pytest.approx(0.75, rel=1e-9)
    assert rated["heat_rate_w"] == pytest.approx(sized["heat_rate_w"])


def test_ntu_and_lmtd_methods_agree():
    n = heat_exchanger_ntu(2000, 4000, 100, 20, ua_w_k=3000, arrangement="counterflow")
    l = heat_exchanger_lmtd(100, n["hot_outlet_c"], 20, n["cold_outlet_c"], arrangement="counterflow",
                            overall_u_w_m2k=1.0, heat_duty_w=n["heat_rate_w"])
    assert l["required_area_m2"] == pytest.approx(3000, rel=1e-9)     # U = 1 so A = UA


def test_parallel_flow_ntu_and_lmtd_methods_agree():
    n = heat_exchanger_ntu(1500, 2500, 120, 30, ua_w_k=2200, arrangement="parallel_flow")
    l = heat_exchanger_lmtd(120, n["hot_outlet_c"], 30, n["cold_outlet_c"], arrangement="parallel_flow",
                            overall_u_w_m2k=1.0, heat_duty_w=n["heat_rate_w"])
    assert l["required_area_m2"] == pytest.approx(2200, rel=1e-9)


def test_shell_and_tube_ntu_and_corrected_lmtd_methods_agree():
    n = heat_exchanger_ntu(2000, 2400, 150, 30, ua_w_k=2500, arrangement="shell_and_tube")
    f = shell_and_tube_correction_factor(150, n["hot_outlet_c"], 30, n["cold_outlet_c"])
    l = heat_exchanger_lmtd(150, n["hot_outlet_c"], 30, n["cold_outlet_c"], correction_factor=f,
                            overall_u_w_m2k=1.0, heat_duty_w=n["heat_rate_w"])
    assert l["required_area_m2"] == pytest.approx(2500, rel=1e-6)


@pytest.mark.parametrize("call", [
    lambda: exchanger_effectiveness(-1, 0.5),
    lambda: exchanger_effectiveness(1, 1.5),
    lambda: exchanger_effectiveness(1, -0.1),
    lambda: exchanger_effectiveness(1, 0.5, "plate"),
    lambda: exchanger_ntu(1.0, 0.5),
    lambda: exchanger_ntu(-0.1, 0.5),
    lambda: heat_exchanger_ntu(0, 100, 100, 20, ua_w_k=10),
    lambda: heat_exchanger_ntu(100, 100, 20, 100, ua_w_k=10),
    lambda: heat_exchanger_ntu(100, 100, 100, 20),
    lambda: heat_exchanger_ntu(100, 100, 100, 20, ua_w_k=10, effectiveness=0.5),
    lambda: heat_exchanger_ntu(100, 100, 100, 20, ua_w_k=0),
    lambda: heat_exchanger_ntu(100, 100, float("inf"), 20, ua_w_k=10),
])
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
