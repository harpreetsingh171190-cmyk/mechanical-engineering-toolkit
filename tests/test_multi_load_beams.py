import pytest
from engineering_toolkit import (beam_multi_load, fixed_fixed_beam_point_load, fixed_fixed_beam_udl,
                                 beam_deflection_point_load, simply_supported_beam_point_load,
                                 simply_supported_beam_udl, cantilever_beam_end_load, cantilever_beam_udl,
                                 parse_numeric_rows)

E, I = 200e9, 8e-6
EI = E * I


# ---- fixed-fixed beams: textbook closed forms -------------------------------------------------

def test_fixed_fixed_central_point_load():
    L, P = 4.0, 1000.0
    r = fixed_fixed_beam_point_load(L, P, L / 2, E, I)
    assert r["reaction_left_n"] == pytest.approx(P / 2) and r["reaction_right_n"] == pytest.approx(P / 2)
    assert r["moment_at_left_nm"] == pytest.approx(-P * L / 8) and r["moment_at_right_nm"] == pytest.approx(-P * L / 8)
    assert r["maximum_moment_nm"] == pytest.approx(P * L / 8)
    assert r["maximum_deflection_m"] == pytest.approx(P * L**3 / (192 * EI))
    assert r["maximum_deflection_location_m"] == pytest.approx(L / 2)
    assert r["deflection_m"][0] == pytest.approx(0, abs=1e-15) and r["deflection_m"][-1] == pytest.approx(0, abs=1e-15)


@pytest.mark.parametrize("a", [0.5, 1.0, 2.5, 3.5])
def test_fixed_fixed_off_centre_point_load(a):
    L, P = 4.0, 1500.0
    b = L - a
    r = fixed_fixed_beam_point_load(L, P, a, E, I)
    assert r["moment_at_left_nm"] == pytest.approx(-P * a * b**2 / L**2)
    assert r["moment_at_right_nm"] == pytest.approx(-P * a**2 * b / L**2)
    assert r["reaction_left_n"] == pytest.approx(P * b**2 * (L + 2 * a) / L**3)
    assert r["reaction_left_n"] + r["reaction_right_n"] == pytest.approx(P)
    # deflection under the load is P a^3 b^3 / (3 E I L^3); the maximum sits in the longer segment
    d_under = P * a**3 * b**3 / (3 * EI * L**3)
    assert _nearest(r, a, "deflection_m") == pytest.approx(d_under)
    assert r["maximum_deflection_m"] >= d_under
    fine = fixed_fixed_beam_point_load(L, P, a, E, I, samples=20001)
    assert r["maximum_deflection_m"] == pytest.approx(max(fine["deflection_m"]), rel=1e-6)
    assert r["maximum_deflection_location_m"] == pytest.approx(fine["x_m"][fine["deflection_m"].index(max(fine["deflection_m"]))], abs=1e-3)
    assert (r["maximum_deflection_location_m"] > a) == (b > a)      # longer segment holds the maximum


def test_fixed_fixed_udl():
    L, w = 5.0, 800.0
    r = fixed_fixed_beam_udl(L, w, E, I, samples=1001)
    assert r["reaction_left_n"] == pytest.approx(w * L / 2)
    assert r["moment_at_left_nm"] == pytest.approx(-w * L**2 / 12) and r["moment_at_right_nm"] == pytest.approx(-w * L**2 / 12)
    assert r["moment_nm"][len(r["x_m"]) // 2] == pytest.approx(w * L**2 / 24)
    assert r["maximum_moment_nm"] == pytest.approx(w * L**2 / 12)
    assert r["maximum_deflection_m"] == pytest.approx(w * L**4 / (384 * EI))


def test_fixed_fixed_is_stiffer_than_simply_supported():
    L, w = 4.0, 1000.0
    ff = fixed_fixed_beam_udl(L, w, E, I)["maximum_deflection_m"]
    ss = simply_supported_beam_udl(L, w, E, I)["maximum_deflection_m"]
    assert ff == pytest.approx(ss / 5)           # 1/384 versus 5/384


def test_fixed_fixed_reactions_do_not_need_E_and_I():
    r = fixed_fixed_beam_point_load(4.0, 1000.0, 1.0)
    assert r["moment_at_left_nm"] == pytest.approx(-1000 * 1 * 9 / 16)
    assert "deflection_m" not in r


def test_fixed_fixed_moment_equilibrium():
    # moments about the left end with sagging-positive end moments: R_B L - M_left + M_right = sum(P a)
    L = 6.0
    r = beam_multi_load(L, [(1.0, 700.0), (4.5, 1200.0)], [(2.0, 5.0, 300.0)], "fixed_fixed")
    load_moment = 700 * 1 + 1200 * 4.5 + 300 * 3 * 3.5
    assert r["reaction_right_n"] * L - r["moment_at_left_nm"] + r["moment_at_right_nm"] == pytest.approx(load_moment)


# ---- multi-load beams: superposition ----------------------------------------------------------

def test_simply_supported_matches_existing_single_load_functions():
    L, P, a = 4.0, 1000.0, 1.0
    multi = beam_multi_load(L, [(a, P)], None, "simply_supported", E, I, samples=401)
    single = simply_supported_beam_point_load(L, P, a, samples=401)
    defl = beam_deflection_point_load(L, P, a, E, I)
    assert multi["reaction_left_n"] == pytest.approx(single["reaction_left_n"])
    assert multi["reaction_right_n"] == pytest.approx(single["reaction_right_n"])
    assert multi["maximum_moment_nm"] == pytest.approx(single["maximum_moment_nm"])
    assert multi["maximum_deflection_m"] == pytest.approx(defl["maximum_deflection_m"], rel=1e-7)
    assert multi["maximum_deflection_location_m"] == pytest.approx(defl["maximum_deflection_location_m"], rel=1e-6)


def test_full_span_udl_matches_existing_function():
    L, w = 6.0, 2000.0
    multi = beam_multi_load(L, None, [(0.0, L, w)], "simply_supported", E, I)
    udl = simply_supported_beam_udl(L, w, E, I)
    assert multi["maximum_moment_nm"] == pytest.approx(w * L**2 / 8)
    assert multi["maximum_moment_location_m"] == pytest.approx(L / 2, rel=1e-6)
    assert multi["maximum_deflection_m"] == pytest.approx(udl["maximum_deflection_m"], rel=1e-9)


def test_cantilever_cases_match_existing_functions():
    L = 3.0
    tip = beam_multi_load(L, [(L, 500.0)], None, "cantilever", E, I)
    ref = cantilever_beam_end_load(L, 500.0, E, I)
    assert tip["reaction_left_n"] == pytest.approx(500.0) and tip["moment_at_left_nm"] == pytest.approx(-500 * L)
    assert tip["maximum_deflection_m"] == pytest.approx(ref["maximum_deflection_m"])
    assert tip["moment_at_right_nm"] == pytest.approx(0, abs=1e-9) and tip["reaction_right_n"] == pytest.approx(0, abs=1e-9)
    udl = beam_multi_load(L, None, [(0.0, L, 800.0)], "cantilever", E, I)
    ref = cantilever_beam_udl(L, 800.0, E, I)
    assert udl["moment_at_left_nm"] == pytest.approx(-800 * L**2 / 2)
    assert udl["maximum_deflection_m"] == pytest.approx(ref["maximum_deflection_m"])


@pytest.mark.parametrize("support", ["simply_supported", "cantilever", "fixed_fixed"])
def test_superposition_of_loads(support):
    L = 5.0
    p1, p2 = [(1.2, 800.0)], [(3.7, 1300.0)]
    u1, u2 = [(0.5, 2.0, 400.0)], [(2.5, 5.0, 250.0)]
    both = beam_multi_load(L, p1 + p2, u1 + u2, support, E, I)
    a = beam_multi_load(L, p1, u1, support, E, I)
    b = beam_multi_load(L, p2, u2, support, E, I)
    for key in ("reaction_left_n", "reaction_right_n", "moment_at_left_nm", "moment_at_right_nm"):
        assert both[key] == pytest.approx(a[key] + b[key], abs=1e-6)
    # the grids differ (each includes its own load positions), so compare at the shared sample points
    for xi, m_both in zip(both["x_m"], both["moment_nm"]):
        ma = _nearest(a, xi, "moment_nm")
        mb = _nearest(b, xi, "moment_nm")
        if ma is not None and mb is not None:
            assert m_both == pytest.approx(ma + mb, abs=1e-6)


def _nearest(result, x, key):
    for xi, v in zip(result["x_m"], result[key]):
        if abs(xi - x) < 1e-12:
            return v
    return None


def test_equilibrium_for_simply_supported_with_mixed_loads():
    L = 10.0
    r = beam_multi_load(L, [(2.0, 1000.0), (4.0, 500.0)], [(0.0, 6.0, 200.0)], "simply_supported")
    assert r["total_load_n"] == pytest.approx(2700.0)
    assert r["reaction_left_n"] + r["reaction_right_n"] == pytest.approx(2700.0)
    # moments about the right support
    assert r["reaction_left_n"] * L == pytest.approx(1000 * 8 + 500 * 6 + 1200 * 7)
    assert abs(r["moment_nm"][0]) < 1e-9 and abs(r["moment_nm"][-1]) < 1e-9


def test_partial_udl_extreme_moment_is_found_between_samples():
    r = beam_multi_load(10.0, None, [(2.0, 6.0, 1000.0)], "simply_supported", samples=11)
    assert r["reaction_left_n"] == pytest.approx(2400.0) and r["reaction_right_n"] == pytest.approx(1600.0)
    assert r["maximum_moment_nm"] == pytest.approx(7680.0, rel=1e-9)
    assert r["maximum_moment_location_m"] == pytest.approx(4.4, rel=1e-6)


def test_shear_area_gives_moment_change():
    r = beam_multi_load(8.0, [(3.0, 900.0)], [(4.0, 8.0, 150.0)], "simply_supported", samples=2001)
    x, v, m = r["x_m"], r["shear_n"], r["moment_nm"]
    area = 0.0
    for i in range(len(x) - 1):
        if abs(x[i + 1] - 3.0) < 1e-12:   # skip the step: shear value at the load is the post-jump value
            continue
        area += (v[i] + v[i + 1]) / 2 * (x[i + 1] - x[i])
    assert area == pytest.approx(m[-1] - m[0], abs=5.0)    # step in shear causes a small trapezoid error


def test_loads_at_the_supports_go_straight_into_reactions():
    r = beam_multi_load(4.0, [(0.0, 1000.0)], None, "simply_supported")
    assert r["reaction_left_n"] == pytest.approx(1000.0) and r["reaction_right_n"] == pytest.approx(0, abs=1e-9)
    assert max(abs(v) for v in r["shear_n"]) == pytest.approx(0, abs=1e-9)
    r = beam_multi_load(4.0, [(4.0, 1000.0)], None, "simply_supported")
    assert r["reaction_right_n"] == pytest.approx(1000.0)
    assert max(abs(v) for v in r["shear_n"]) == pytest.approx(0, abs=1e-9)


def test_upward_loads_are_allowed_as_negative_values():
    r = beam_multi_load(4.0, [(1.0, 1000.0), (3.0, -1000.0)], None, "simply_supported")
    assert r["total_load_n"] == pytest.approx(0, abs=1e-12)
    assert r["reaction_left_n"] == pytest.approx(500.0) and r["reaction_right_n"] == pytest.approx(-500.0)


def test_no_deflection_without_E_and_I():
    r = beam_multi_load(4.0, [(2.0, 100.0)], None, "cantilever")
    assert "deflection_m" not in r and "maximum_deflection_m" not in r


def test_parse_numeric_rows():
    assert parse_numeric_rows("1, 2; 3 4\n5,6", 2) == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    assert parse_numeric_rows("   ", 3) == []
    for bad in ("1, 2, 3", "a, b", "1, nan"):
        with pytest.raises(ValueError):
            parse_numeric_rows(bad, 2)


@pytest.mark.parametrize("call", [
    lambda: beam_multi_load(0, [(1, 100)]),
    lambda: beam_multi_load(4, None, None),
    lambda: beam_multi_load(4, [(5, 100)]),
    lambda: beam_multi_load(4, [(-1, 100)]),
    lambda: beam_multi_load(4, [(1, float("nan"))]),
    lambda: beam_multi_load(4, [(1,)]),
    lambda: beam_multi_load(4, None, [(3, 2, 100)]),
    lambda: beam_multi_load(4, None, [(0, 5, 100)]),
    lambda: beam_multi_load(4, [(1, 100)], support="propped"),
    lambda: beam_multi_load(4, [(1, 100)], youngs_modulus_pa=E),
    lambda: beam_multi_load(4, [(1, 100)], samples=1),
    lambda: fixed_fixed_beam_point_load(4, 100, 5),
    lambda: fixed_fixed_beam_point_load(4, -100, 1),
    lambda: fixed_fixed_beam_udl(4, -1),
])
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
