"""Ideal projectile motion (no air resistance, launch and landing at the same height).

Run as a script:

    python projectile_motion.py --speed 40 --angles 30 45 60
    python projectile_motion.py --speed 40 --save trajectories.png
"""
from __future__ import annotations

import argparse
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

G_STANDARD = 9.81  # m/s^2


def _validate(speed: float, angle_deg: float, gravity: float) -> None:
    vals = np.asarray([speed, angle_deg, gravity], dtype=float)
    if not np.all(np.isfinite(vals)):
        raise ValueError("Speed, angle and gravity must be finite.")
    if speed < 0:
        raise ValueError("Speed must be non-negative.")
    if gravity <= 0:
        raise ValueError("Gravity must be positive.")


def projectile_trajectory(speed: float, angle_deg: float, *, gravity: float = G_STANDARD,
                          samples: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, horizontal position and vertical position arrays (SI units)."""
    _validate(speed, angle_deg, gravity)
    if not isinstance(samples, (int, np.integer)) or isinstance(samples, bool) or samples < 2:
        raise ValueError("samples must be an integer of at least 2.")
    angle = np.deg2rad(angle_deg)
    vx, vy = speed * np.cos(angle), speed * np.sin(angle)
    flight_time = max(0.0, 2.0 * vy / gravity)
    t = np.linspace(0.0, flight_time, int(samples))
    x = vx * t
    y = vy * t - 0.5 * gravity * t**2
    y[-1] = 0.0  # remove tiny floating-point residual at the landing point
    return t, x, y


def trajectory_summary(speed: float, angle_deg: float, *, gravity: float = G_STANDARD) -> dict[str, float]:
    """Closed-form flight time, range and maximum height."""
    _validate(speed, angle_deg, gravity)
    angle = np.deg2rad(angle_deg)
    vy = speed * np.sin(angle)
    return {
        "flight_time_s": max(0.0, 2.0 * vy / gravity),
        "range_m": max(0.0, speed**2 * np.sin(2.0 * angle) / gravity),
        "max_height_m": max(0.0, vy**2 / (2.0 * gravity)),
    }


def plot_trajectories(speed: float = 40.0, angles: Iterable[float] = (45.0, 60.0), *,
                      gravity: float = G_STANDARD, ax=None):
    """Plot one or more launch angles and return the Axes for reuse/customization."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.5))
    for angle in angles:
        _, x, y = projectile_trajectory(speed, float(angle), gravity=gravity)
        ax.plot(x, y, label=fr"$\theta={angle:g}^\circ$ (range {x[-1]:.1f} m)")
    ax.set(title="Projectile motion (no air resistance)",
           xlabel="Horizontal distance (m)", ylabel="Height (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.figure.tight_layout()
    return ax


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--speed", type=float, default=40.0, help="launch speed in m/s")
    parser.add_argument("--angles", type=float, nargs="+", default=[45.0, 60.0], help="launch angles in degrees")
    parser.add_argument("--gravity", type=float, default=G_STANDARD, help="gravity in m/s^2")
    parser.add_argument("--save", metavar="FILE", help="save the plot instead of showing it")
    args = parser.parse_args(argv)

    for angle in args.angles:
        s = trajectory_summary(args.speed, angle, gravity=args.gravity)
        print(f"{angle:g} deg: time {s['flight_time_s']:.2f} s, "
              f"range {s['range_m']:.1f} m, max height {s['max_height_m']:.1f} m")
    plot_trajectories(args.speed, args.angles, gravity=args.gravity)
    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Saved plot to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
