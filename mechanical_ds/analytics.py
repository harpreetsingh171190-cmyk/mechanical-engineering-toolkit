"""EDA, anomaly detection, forecasting and engineering simulations."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def load_sensor_csv(path: str) -> pd.DataFrame:
    """Load CSV and validate that it contains at least one numeric measurement."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV contains no rows")
    if not any(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
        raise ValueError("CSV must contain at least one numeric column")
    return df


def eda_summary(data: pd.DataFrame) -> dict:
    """Return compact EDA output suitable for dashboards/reports."""
    if data.empty:
        raise ValueError("data must not be empty")
    numeric = data.select_dtypes(include="number")
    return {"rows": int(len(data)), "columns": int(len(data.columns)),
            "missing_by_column": data.isna().sum().astype(int).to_dict(),
            "describe": numeric.describe().replace({np.nan: None}).to_dict(),
            "correlation": numeric.corr().replace({np.nan: None}).to_dict()}


def detect_anomalies(values, window: int = 7, z_threshold: float = 3.5) -> np.ndarray:
    """Robust median/MAD anomaly flags; handles constant series and finite values."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("values must be a non-empty 1-D array of finite numbers")
    if window < 3 or not math.isfinite(z_threshold) or z_threshold <= 0:
        raise ValueError("window must be >= 3 and threshold must be positive")
    s = pd.Series(x)
    med = s.rolling(window, center=True, min_periods=1).median().to_numpy()
    residual = np.abs(x - med)
    mad = np.median(residual)
    if mad == 0:
        # Fallback to global robust scale, still flags isolated spikes in flat series.
        mad = np.median(np.abs(x - np.median(x)))
    if mad == 0:
        return residual > 0
    return residual / (1.4826 * mad) > z_threshold


def moving_average_forecast(values, horizon: int = 1, window: int = 5) -> np.ndarray:
    """Baseline forecast repeating the mean of the most recent `window` points."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("values must be a non-empty 1-D array of finite numbers")
    if not isinstance(horizon, int) or horizon < 1 or not isinstance(window, int) or window < 1:
        raise ValueError("horizon and window must be positive integers")
    return np.full(horizon, float(np.mean(x[-window:])))


def forecast_metrics(actual, predicted) -> dict[str, float]:
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if a.shape != p.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("actual and predicted must be same-sized non-empty finite arrays")
    return {"mae": float(np.mean(np.abs(a-p))), "rmse": float(np.sqrt(np.mean((a-p)**2)))}


def simulate_mass_spring(mass_kg: float, stiffness_n_m: float, damping_n_s_m: float,
                         initial_displacement_m: float, initial_velocity_m_s: float,
                         duration_s: float = 10.0, samples: int = 1001) -> dict[str, np.ndarray]:
    """Numerically integrate m*x''+c*x'+k*x=0 using SciPy solve_ivp."""
    from scipy.integrate import solve_ivp
    vals = [mass_kg, stiffness_n_m, damping_n_s_m, duration_s]
    if not all(math.isfinite(float(v)) for v in vals) or mass_kg <= 0 or stiffness_n_m <= 0 or damping_n_s_m < 0 or duration_s <= 0:
        raise ValueError("mass, stiffness and duration must be positive; damping non-negative")
    if samples < 2:
        raise ValueError("samples must be >= 2")
    t = np.linspace(0, duration_s, samples)
    def rhs(_, y): return [y[1], -(damping_n_s_m*y[1] + stiffness_n_m*y[0])/mass_kg]
    sol = solve_ivp(rhs, (0, duration_s), [initial_displacement_m, initial_velocity_m_s], t_eval=t, rtol=1e-8, atol=1e-10)
    if not sol.success: raise RuntimeError(sol.message)
    return {"time_s": sol.t, "displacement_m": sol.y[0], "velocity_m_s": sol.y[1]}


def optimize_beam_depth(load_n: float, span_m: float, width_m: float,
                        allowable_stress_pa: float, material_cost_per_m3: float = 1.0) -> dict[str, float]:
    """Size a rectangular beam: smallest depth that meets the bending-stress limit.

    Uses a simply-supported central point load, sigma_max = 3*P*L / (2*b*h^2),
    solved in closed form for h (volume is monotonic in depth, so the minimum
    feasible depth is also the minimum-material design). A 1 micrometre floor
    avoids a zero-depth result.
    This simplified model omits deflection, buckling, fatigue and design-code factors.
    """
    vals = [load_n, span_m, width_m, allowable_stress_pa, material_cost_per_m3]
    if not all(math.isfinite(float(v)) and float(v) > 0 for v in vals):
        raise ValueError("all inputs must be finite and positive")
    required_h = math.sqrt(3*load_n*span_m/(2*width_m*allowable_stress_pa))
    depth = max(required_h, 1e-6)
    volume = width_m * depth * span_m
    return {"depth_m": depth, "volume_m3": volume, "estimated_cost": volume*material_cost_per_m3,
            "calculated_stress_pa": 3*load_n*span_m/(2*width_m*depth**2)}
