import numpy as np
import pandas as pd

# Seamless GPU acceleration via RAPIDS cuDF when CUDA is available; falls back to pandas
# try:
#     import cudf as df_lib
# except ImportError:
#     df_lib = pd

df_lib = pd


def compute_rolling_velocities(
    df: pd.DataFrame,
    windows: tuple[str, ...] = ("1h", "24h", "7D"),
) -> pd.DataFrame:
    """
    Computes strictly causal rolling window spend velocities and burst z-scores:
    V_u^{(t, Delta_t)} = sum_{tau in T_u} x_amt^{(u, tau)} * I(t - Delta_t <= tau < t)
    Uses closed='left' so the current transaction never leaks into its own feature vector.
    """
    work = df.sort_values(["src", "timestamp"]).copy()
    work["dt_index"] = pd.to_datetime(work["timestamp"], unit="s")
    work = work.set_index("dt_index")

    grouped = work.groupby("src")["amount"]
    for w in windows:
        roll_sum = (
            grouped.rolling(w, closed="left")
            .sum()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )
        roll_cnt = (
            grouped.rolling(w, closed="left")
            .count()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )
        roll_mean = (
            grouped.rolling(w, closed="left")
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )
        roll_std = (
            grouped.rolling(w, closed="left")
            .std()
            .reset_index(level=0, drop=True)
            .fillna(1.0)
        )

        work[f"vel_amt_{w}"] = roll_sum.values
        work[f"vel_cnt_{w}"] = roll_cnt.values
        work[f"vel_zscore_{w}"] = (work["amount"].values - roll_mean.values) / (
            roll_std.values + 1e-6
        )

    return work.reset_index(drop=True).sort_index()


def bochner_time_encoding(delta_t: np.ndarray, dim: int = 8) -> np.ndarray:
    """
    Vectorized Bochner harmonic time-delta encoding:
    Phi(Delta_t) = sqrt(1/d) * [cos(omega_1 * Delta_t + phi_1), ..., cos(omega_d * Delta_t + phi_d)]
    """
    omega = np.logspace(0, -4, num=dim)
    phi = np.linspace(0.0, np.pi, num=dim)
    dt_col = np.asarray(delta_t, dtype=np.float64).reshape(-1, 1)
    return np.sqrt(1.0 / dim) * np.cos(dt_col * omega + phi)
