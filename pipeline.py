import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600, show_spinner="Завантаження телеметрії...")
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=3600)
def clean_data(df: pd.DataFrame, col: str, k: float = 1.5) -> tuple[pd.DataFrame, dict]:
    report = {"rows_in": len(df)}
    df_clean = df.drop_duplicates(subset=["timestamp"]).copy()
    report["duplicates"] = report["rows_in"] - len(df_clean)
    report["missing_before"] = int(df_clean[col].isna().sum())

    # Етап 1: первинне виявлення викидів за (1)-(2) та обробка розривів (>3 відліки не інтерполюються)
    q1, q3 = df_clean[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - k * iqr, q3 + k * iqr
    is_outlier = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
    report["outliers"] = int(is_outlier.sum())

    s_masked = df_clean[col].mask(is_outlier, np.nan)
    na = s_masked.isna()
    run_len = na.groupby((na != na.shift()).cumsum()).transform("sum")

    # Етап 2: лінійна інтерполяція внутрішніх розривів до 3 відліків; розриви >3 відліків вилучаються
    df_clean[col] = s_masked.mask(na & (run_len > 3)).interpolate(method="linear", limit_area="inside")
    df_clean = df_clean.dropna(subset=[col]).reset_index(drop=True)
    report["rows_out"] = len(df_clean)
    return df_clean, report


@st.cache_data(ttl=3600)
def aggregate_series(df: pd.DataFrame, col: str, rule: str = "1h") -> pd.DataFrame:
    # Побудова регулярної часової сітки з лівим вирівнюванням інтервалів
    agg_df = (
        df.set_index("timestamp")[col]
        .resample(rule, closed="left", label="left")
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    return agg_df
