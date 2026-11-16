import pandas as pd


def strict_oot_maturity_split(
    events_df: pd.DataFrame,
    t_split: float,
    t_val_end: float,
    maturity_lag_sec: float = 30.0 * 86400.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enforces strict Out-of-Time (OOT) validation splits with chargeback maturity lag Delta_label:
    D_train = {x_i | t_i <= T_split - Delta_label}
    D_val   = {x_i | T_split < t_i <= T_val}
    """
    train_cutoff = t_split - maturity_lag_sec
    d_train = events_df[events_df["timestamp"] <= train_cutoff].copy()
    d_val = events_df[(events_df["timestamp"] > t_split) & (events_df["timestamp"] <= t_val_end)].copy()
    return d_train, d_val