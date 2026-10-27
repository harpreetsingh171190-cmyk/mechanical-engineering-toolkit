import math
import pytest
from engineering_toolkit import (beam_multi_load, continuous_beam, fixed_fixed_beam_udl,
                                 propped_cantilever_point_load, propped_cantilever_udl,
                                 simply_supported_beam_udl)

E, I = 200e9, 2000e-8
EI = E * I


def test_propped_cantilever_udl_textbook_values():
    w, L = 1000.0, 4.0
    r = propped_cantilever_udl(L, w, E, I)
    assert r["reaction_right_n"] == pytest.approx(3 * w * L / 8)
    assert r["reaction_left_n"] == pytest.approx(5 * w * L / 8)
    assert r["moment_at_left_nm"] == pytest.approx(-w * L**2 / 8)
    assert r["moment_at_right_nm"] == pytest.approx(0.0, abs=1e-6)
    assert r["maximum_deflection_m"] == pytest.approx(w * L**4 / (185 * EI), rel=0.01)
    assert r["maximum_deflection_location_m"] == pytest.approx(0.5785 * L, rel=0.01)   # 0.4215 L from the prop


def test_propped_cantilever_midspan_point_load():
    P, L = 1000.0, 4.0
    r = propped_cantilever_point_load(L, P, L / 2, E, I)
    assert r["reaction_right_n"] == pytest.approx(5 * P / 16)
    assert r["reaction_left_n"] == pytest.approx(11 * P / 16)
    assert r["moment_at_left_nm"] == pytest.approx(-3 * P * L / 16)
    assert r["maximum_moment_nm"] == pytest.approx(3 * P * L / 16)
    # largest sagging moment sits under the load: 5PL/32
    m_mid = r["moment_nm"][r["x_m"].index(L / 2)]
    assert m_mid == pytest.approx(5 * P * L / 32)


def test_propped_cantilever_general_formula_and_multi_load():
    P, L, a = 800.0, 5.0, 2.0
    b = L - a
    r = propped_cantilever_point_load(L, P, a)
    assert r["reaction_right_n"] == pytest.approx(P * a**2 * (3 * L - a) / (2 * L**3))
    assert r["moment_at_left_nm"] == pytest.approx(-P * a * b * (L + b) / (2 * L**2))
    mixed = beam_multi_load(L, [(a, P)], [(0, L, 300.0)], "propped_cantilever")
    assert mixed["reaction_left_n"] + mixed["reaction_right_n"] == pytest.approx(P + 300.0 * L)
    assert mixed["moment_at_right_nm"] == pytest.approx(0.0, abs=1e-6)


def test_two_equal_spans_under_uniform_load():
    w, L = 1000.0, 4.0
    r = continuous_beam([L, L], None, [(0, 2 * L, w)], youngs_modulus_pa=E, second_moment_m4=I)
    assert r["reactions_n"] == pytest.approx([3 * w * L / 8, 5 * w * L / 4, 3 * w * L / 8])
    assert r["moments_at_supports_nm"][1] == pytest.approx(-w * L**2 / 8)
    assert r["maximum_sagging_moment_nm"] == pytest.approx(9 * w * L**2 / 128)
    assert r["maximum_sagging_moment_location_m"] == pytest.approx(3 * L / 8)
    assert r["maximum_hogging_moment_nm"] == pytest.approx(-w * L**2 / 8)
    assert r["maximum_deflection_m"] == pytest.approx(w * L**4 / (185 * EI), rel=0.01)


def test_three_equal_spans_under_uniform_load():
    w, L = 1000.0, 3.0
    r = continuous_beam([L, L, L], None, [(0, 3 * L, w)])
    assert r["reactions_n"] == pytest.approx([0.4 * w * L, 1.1 * w * L, 1.1 * w * L, 0.4 * w * L])
    assert r["moments_at_supports_nm"][1] == pytest.approx(-w * L**2 / 10)
    assert r["moments_at_supports_nm"][2] == pytest.approx(-w * L**2 / 10)


def test_two_spans_with_a_midspan_point_load_on_one_span():
    P, L = 1000.0, 4.0
    r = continuous_beam([L, L], [(L / 2, P)])
    assert r["reactions_n"] == pytest.approx([13 * P / 32, 11 * P / 16, -3 * P / 32])
    assert r["moments_at_supports_nm"][1] == pytest.approx(-3 * P * L / 32)


def test_force_and_moment_equilibrium_for_mixed_loads_and_a_load_over_a_support():
    pts = [(1.0, 2000.0), (3.0, 1500.0), (8.0, -500.0)]      # the 3 m load sits exactly on a support
    udl = [(2.0, 6.0, 800.0), (7.0, 9.5, 300.0)]
    r = continuous_beam([3.0, 4.0, 2.5], pts, udl)
    assert sum(r["reactions_n"]) == pytest.approx(r["total_load_n"])
    about_left = sum(rx * x for rx, x in zip(r["reactions_n"], r["support_positions_m"]))
    applied = sum(p * x for x, p in pts) + sum(w * (e * e - s0 * s0) / 2 for s0, e, w in udl)
    assert about_left == pytest.approx(applied)
    fixed = continuous_beam([3.0, 4.0, 2.5], pts, udl, left_end="fixed")
    assert sum(fixed["reactions_n"]) == pytest.approx(fixed["total_load_n"])


def test_single_span_end_conditions_match_the_closed_form_beams():
    w, L = 750.0, 5.0
    ss = continuous_beam([L], None, [(0, L, w)], youngs_modulus_pa=E, second_moment_m4=I)
    ref = simply_supported_beam_udl(L, w, E, I)
    assert ss["maximum_moment_nm"] == pytest.approx(ref["maximum_moment_nm"])
    assert ss["maximum_deflection_m"] == pytest.approx(5 * w * L**4 / (384 * EI), rel=1e-4)
    ff = continuous_beam([L], None, [(0, L, w)], left_end="fixed", right_end="fixed", youngs_modulus_pa=E, second_moment_m4=I)
    ref_ff = fixed_fixed_beam_udl(L, w, E, I)
    assert ff["moment_at_left_nm"] == pytest.approx(ref_ff["moment_at_left_nm"])
    assert ff["maximum_deflection_m"] == pytest.approx(ref_ff["maximum_deflection_m"], rel=1e-4)
    pp = continuous_beam([L], [(2.0, 900.0)], None, left_end="fixed", right_end="pinned")
    ref_pp = propped_cantilever_point_load(L, 900.0, 2.0)
    assert pp["reactions_n"][0] == pytest.approx(ref_pp["reaction_left_n"])
    assert pp["moment_at_left_nm"] == pytest.approx(ref_pp["moment_at_left_nm"])


def test_overhang_with_a_free_end():
    r = continuous_beam([4.0, 2.0], [(6.0, 1000.0)], left_end="pinned", right_end="free")
    assert r["reactions_n"][0] == pytest.approx(-500.0) and r["reactions_n"][1] == pytest.approx(1500.0)
    assert r["moments_at_supports_nm"][1] == pytest.approx(-2000.0)
    assert r["moment_at_right_nm"] == pytest.approx(0.0, abs=1e-6)


def test_span_stiffness_enters_through_the_three_moment_equation():
    w, l1, l2, i1, i2 = 1000.0, 3.0, 5.0, 2.0, 1.0
    r = continuous_beam([l1, l2], None, [(0, l1 + l2, w)], second_moment_m4=[i1, i2])
    expected = -w * (l1**3 / i1 + l2**3 / i2) / (8 * (l1 / i1 + l2 / i2))
    assert r["moments_at_supports_nm"][1] == pytest.approx(expected)
    equal = continuous_beam([l1, l2], None, [(0, l1 + l2, w)])["moments_at_supports_nm"][1]
    assert equal == pytest.approx(-w * (l1**3 + l2**3) / (8 * (l1 + l2)))
    assert expected != pytest.approx(equal)


def test_relative_stiffness_does_not_change_forces_for_equal_spans():
    a = continuous_beam([3.0, 3.0], None, [(0, 6, 500)])
    b = continuous_beam([3.0, 3.0], None, [(0, 6, 500)], second_moment_m4=7e-5)
    assert a["reactions_n"] == pytest.approx(b["reactions_n"]) and "deflection_m" not in b


def test_shear_and_moment_arrays_are_consistent():
    w, L = 1000.0, 4.0
    r = continuous_beam([L, L], None, [(0, 2 * L, w)])
    assert r["shear_n"][0] == pytest.approx(3 * w * L / 8)
    i = r["x_m"].index(L)
    assert r["shear_n"][i] == pytest.approx(3 * w * L / 8 - w * L + 5 * w * L / 4)   # just right of the middle support
    assert r["moment_nm"][i] == pytest.approx(-w * L**2 / 8)
    assert r["maximum_shear_n"] == pytest.approx(5 * w * L / 8)


@pytest.mark.parametrize("call", [
    lambda: continuous_beam([], None, [(0, 1, 1)]),
    lambda: continuous_beam([3, -1], None, [(0, 1, 1)]),
    lambda: continuous_beam([3, 3]),
    lambda: continuous_beam([3, 3], [(7.0, 100)]),
    lambda: continuous_beam([3, 3], None, [(0, 7, 100)]),
    lambda: continuous_beam([3, 3], [(1, 100)], left_end="roller"),
    lambda: continuous_beam([3], [(1, 100)], left_end="free", right_end="free"),
    lambda: continuous_beam([3], [(1, 100)], left_end="pinned", right_end="free"),
    lambda: continuous_beam([3, 3], [(1, 100)], second_moment_m4=[1e-5]),
    lambda: continuous_beam([3, 3], [(1, 100)], youngs_modulus_pa=E),
    lambda: continuous_beam([3, 3], [(1, float("nan"))]),
    lambda: continuous_beam([3, 3], [(1, 100)], samples=1),
    lambda: propped_cantilever_point_load(4, 1000, 5),
    lambda: propped_cantilever_udl(0, 100),
])
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
