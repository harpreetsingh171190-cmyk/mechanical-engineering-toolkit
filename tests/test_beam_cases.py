import pytest
from engineering_toolkit import (simply_supported_beam_udl, cantilever_beam_end_load, cantilever_beam_udl,
                                 beam_deflection_point_load)

E, I = 200e9, 8e-6


def test_ss_udl_reactions_moment_and_deflection():
    L, w = 6.0, 2000.0
    r = simply_supported_beam_udl(L, w, E, I, samples=601)
    assert r["reaction_left_n"] == pytest.approx(w * L / 2) and r["reaction_right_n"] == pytest.approx(w * L / 2)
    assert r["maximum_moment_nm"] == pytest.approx(w * L**2 / 8)
    assert max(r["moment_nm"]) == pytest.approx(w * L**2 / 8)
    assert r["shear_n"][0] == pytest.approx(w * L / 2) and r["shear_n"][-1] == pytest.approx(-w * L / 2)
    assert r["maximum_deflection_m"] == pytest.approx(5 * w * L**4 / (384 * E * I))
    assert max(r["deflection_m"]) == pytest.approx(r["maximum_deflection_m"], rel=1e-9)
    assert r["deflection_m"][0] == pytest.approx(0, abs=1e-15) and r["deflection_m"][-1] == pytest.approx(0, abs=1e-12)


def test_ss_udl_equals_sum_of_many_point_loads_in_the_limit():
    # statics check: area under the shear diagram from 0 to midspan equals the midspan moment
    L, w = 4.0, 1000.0
    r = simply_supported_beam_udl(L, w, samples=4001)
    mid = len(r["x_m"]) // 2
    area = sum((r["shear_n"][i] + r["shear_n"][i + 1]) / 2 * (r["x_m"][i + 1] - r["x_m"][i]) for i in range(mid))
    assert area == pytest.approx(r["moment_nm"][mid], rel=1e-6)


def test_cantilever_end_load():
    L, P = 2.0, 500.0
    r = cantilever_beam_end_load(L, P, E, I)
    assert r["reaction_force_n"] == P and r["fixed_end_moment_nm"] == pytest.approx(P * L)
    assert r["moment_nm"][0] == pytest.approx(-P * L) and r["moment_nm"][-1] == pytest.approx(0, abs=1e-12)
    assert r["maximum_deflection_m"] == pytest.approx(P * L**3 / (3 * E * I))
    assert r["deflection_m"][-1] == pytest.approx(r["maximum_deflection_m"])
    assert r["deflection_m"][0] == 0


def test_cantilever_udl():
    L, w = 3.0, 800.0
    r = cantilever_beam_udl(L, w, E, I)
    assert r["reaction_force_n"] == pytest.approx(w * L)
    assert r["fixed_end_moment_nm"] == pytest.approx(w * L**2 / 2)
    assert r["moment_nm"][0] == pytest.approx(-w * L**2 / 2)
    assert r["maximum_deflection_m"] == pytest.approx(w * L**4 / (8 * E * I))
    assert r["deflection_m"][-1] == pytest.approx(r["maximum_deflection_m"])
    assert r["shear_n"][0] == pytest.approx(w * L) and r["shear_n"][-1] == pytest.approx(0, abs=1e-12)


def test_cantilever_udl_is_stiffer_than_same_total_load_at_tip():
    L, total = 2.0, 1000.0
    udl = cantilever_beam_udl(L, total / L, E, I)["maximum_deflection_m"]
    tip = cantilever_beam_end_load(L, total, E, I)["maximum_deflection_m"]
    assert udl == pytest.approx(tip * 3 / 8)


def test_cantilever_matches_half_of_a_symmetric_ss_beam():
    # a simply supported beam with midspan load P behaves like two cantilevers of length L/2 loaded with P/2
    L, P = 4.0, 1000.0
    ss = beam_deflection_point_load(L, P, L / 2, E, I)["maximum_deflection_m"]
    cant = cantilever_beam_end_load(L / 2, P / 2, E, I)["maximum_deflection_m"]
    assert ss == pytest.approx(cant)


def test_no_deflection_without_E_and_I():
    r = cantilever_beam_end_load(2.0, 100.0)
    assert "deflection_m" not in r and "maximum_deflection_m" not in r


@pytest.mark.parametrize("call", [
    lambda: simply_supported_beam_udl(0, 100),
    lambda: simply_supported_beam_udl(4, -1),
    lambda: cantilever_beam_end_load(2, 100, E, None),
    lambda: cantilever_beam_udl(2, 100, None, I),
    lambda: cantilever_beam_udl(2, 100, E, 0),
    lambda: cantilever_beam_end_load(2, 100, samples=1),
])
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
