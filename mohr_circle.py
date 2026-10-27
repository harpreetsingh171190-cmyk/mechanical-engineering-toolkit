"""Mohr's circle utilities for a 2-D plane-stress state.

Stress units must be consistent (for example MPa throughout). The sign convention
for shear stress follows the standard transformation equations used below.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MohrsCircle:
    """Calculate principal stresses and plot Mohr's circle for plane stress."""

    sigmax: float
    sigmay: float
    tauxy: float

    def __post_init__(self) -> None:
        values = np.asarray([self.sigmax, self.sigmay, self.tauxy], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Stress inputs must be finite numbers.")
        object.__setattr__(self, "sigmax", float(self.sigmax))
        object.__setattr__(self, "sigmay", float(self.sigmay))
        object.__setattr__(self, "tauxy", float(self.tauxy))

    def circleCenter(self) -> float:
        """Return average normal stress, the circle's horizontal center."""
        return (self.sigmax + self.sigmay) / 2.0

    def radius(self) -> float:
        """Return Mohr-circle radius (maximum in-plane shear-stress magnitude)."""
        return float(np.hypot((self.sigmax - self.sigmay) / 2.0, self.tauxy))

    def maximumPrincipalStress(self) -> float:
        return self.circleCenter() + self.radius()

    def minimumPrincipalStress(self) -> float:
        return self.circleCenter() - self.radius()

    def maximumShearStress(self) -> float:
        return self.radius()

    def principalAngle(self, *, degrees: bool = True) -> float:
        """Angle from x-axis to the maximum-principal-stress plane.

        Uses theta_p = 0.5 atan2(2*tau_xy, sigma_x-sigma_y). This is the
        physical plane angle, not the doubled angle shown on Mohr's circle.
        """
        angle = 0.5 * np.arctan2(2.0 * self.tauxy, self.sigmax - self.sigmay)
        return float(np.degrees(angle) if degrees else angle)

    def mohrsCirclePlot(self, ax=None):
        """Plot the circle and return its Axes; creates a figure if needed."""
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        center, radius = self.circleCenter(), self.radius()
        radians = np.linspace(0.0, 2.0 * np.pi, 500)
        sigma = center + radius * np.cos(radians)
        tau = radius * np.sin(radians)
        ax.fill(sigma, tau, color="tab:blue", alpha=0.10)
        ax.plot(sigma, tau, color="black", label="Mohr's circle")
        ax.plot([self.sigmax, self.sigmay], [self.tauxy, -self.tauxy], "o-",
                color="tab:red", label="Stress states")
        ax.plot(center, 0, "ko", label="Center")
        ax.plot([self.minimumPrincipalStress(), self.maximumPrincipalStress()], [0, 0], "ko")
        ax.annotate(r"$\sigma_1$", (self.maximumPrincipalStress(), 0), xytext=(4, 8), textcoords="offset points")
        ax.annotate(r"$\sigma_2$", (self.minimumPrincipalStress(), 0), xytext=(-4, 8), textcoords="offset points", ha="right")
        ax.annotate(r"$\tau_{max}$", (center, radius), xytext=(4, -14), textcoords="offset points")
        ax.set(title="Mohr's Circle", xlabel=r"Normal stress, $\sigma$", ylabel=r"Shear stress, $\tau$")
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.axvline(0, color="0.35", linewidth=0.8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.figure.tight_layout()
        return ax


if __name__ == "__main__":
    try:
        sx = float(input("Normal stress sigma_x (consistent units): "))
        sy = float(input("Normal stress sigma_y (same units): "))
        txy = float(input("Shear stress tau_xy (same units): "))
        circle = MohrsCircle(sx, sy, txy)
        print(f"Principal stress sigma_1: {circle.maximumPrincipalStress():.3f}")
        print(f"Principal stress sigma_2: {circle.minimumPrincipalStress():.3f}")
        print(f"Maximum in-plane shear: {circle.maximumShearStress():.3f}")
        print(f"Average normal stress: {circle.circleCenter():.3f}")
        print(f"Principal-plane angle: {circle.principalAngle():.3f} degrees")
        circle.mohrsCirclePlot()
        plt.show()
    except ValueError as exc:
        raise SystemExit(f"Invalid input: {exc}") from exc
