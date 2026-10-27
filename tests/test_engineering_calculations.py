import numpy as np
import pytest

from mohr_circle import MohrsCircle
from projectile_motion import projectile_trajectory


def test_mohr_circle_uniaxial_stress():
    c = MohrsCircle(100, 0, 0)
    assert c.circleCenter() == pytest.approx(50)
    assert c.radius() == pytest.approx(50)
    assert c.maximumPrincipalStress() == pytest.approx(100)
    assert c.minimumPrincipalStress() == pytest.approx(0)
    assert c.maximumShearStress() == pytest.approx(50)


def test_mohr_circle_pure_shear():
    c = MohrsCircle(0, 0, 30)
    assert c.maximumPrincipalStress() == pytest.approx(30)
    assert c.minimumPrincipalStress() == pytest.approx(-30)
    assert abs(c.principalAngle()) == pytest.approx(45)


def test_mohr_circle_rejects_nonfinite_input():
    with pytest.raises(ValueError):
        MohrsCircle(float("nan"), 0, 0)


def test_projectile_landing_and_range():
    t, x, y = projectile_trajectory(20, 45, samples=101)
    assert len(t) == len(x) == len(y) == 101
    assert y[0] == pytest.approx(0)
    assert y[-1] == pytest.approx(0)
    assert x[-1] == pytest.approx(20**2 / 9.81)
    assert y.max() == pytest.approx((20 * np.sin(np.pi / 4))**2 / (2 * 9.81), rel=1e-3)


def test_projectile_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        projectile_trajectory(-1, 45)
    with pytest.raises(ValueError):
        projectile_trajectory(10, 45, gravity=0)


def test_trajectory_summary_matches_numeric_trajectory():
    from projectile_motion import trajectory_summary
    s = trajectory_summary(20, 45)
    t, x, y = projectile_trajectory(20, 45, samples=2001)
    assert s["range_m"] == pytest.approx(x[-1])
    assert s["flight_time_s"] == pytest.approx(t[-1])
    assert s["max_height_m"] == pytest.approx(y.max(), rel=1e-4)
