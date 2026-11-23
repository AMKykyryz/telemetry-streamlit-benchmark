import gc
import os
import threading
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st
from pipeline import aggregate_series, clean_data, load_data


def generate_telemetry(n_rows: int = 1_000_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-10-01 00:00:00", periods=n_rows, freq="1s", tz="UTC")
    t_sec = np.arange(n_rows, dtype=np.float64)
    diurnal = np.sin(2.0 * np.pi * t_sec / 86400.0)

    latency = 18.0 + 6.5 * diurnal + rng.lognormal(mean=0.4, sigma=0.35, size=n_rows)
    throughput = np.clip(620.0 + 180.0 * diurnal + rng.normal(0.0, 22.0, size=n_rows), 50.0, 1000.0)
    packet_loss = np.clip(0.05 + 0.03 * np.maximum(diurnal, 0.0) + rng.exponential(0.04, size=n_rows), 0.0, 5.0)

    # Внесення 1% викидів (+8 * SD) та 2% випадкових пропусків (NaN)
    outlier_idx = rng.choice(n_rows, size=int(0.01 * n_rows), replace=False)
    latency[outlier_idx] += 8.0 * np.std(latency)
    missing_idx = rng.choice(n_rows, size=int(0.02 * n_rows), replace=False)
    latency[missing_idx] = np.nan

    return pd.DataFrame({
        "timestamp": ts,
        "latency_ms": latency,
        "throughput_mbps": throughput,
        "packet_loss_pct": packet_loss,
    })


class PeakRSSMonitor:
    """Моніторинг пікового RSS із кроком 50 мс у фоновому потоці (правка №22)."""
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.proc = psutil.Process(os.getpid())
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            if rss > self.peak_bytes:
                self.peak_bytes = rss
            time.sleep(self.interval_s)

    def __enter__(self):
        self.peak_bytes = self.proc.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        self._thread.join()

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 ** 2)


def run_single_pass(csv_path: str, use_cache: bool, use_resample: bool) -> dict:
    gc.collect()
    with PeakRSSMonitor(0.05) as mon:
        t0 = time.perf_counter()
        if use_cache:
            df = load_data(csv_path)
            df_clean, _ = clean_data(df, "latency_ms", k=1.5)
            df_plot = aggregate_series(df_clean, "latency_ms", rule="1h") if use_resample else df_clean
        else:
            df = load_data.__wrapped__(csv_path)
            df_clean, _ = clean_data.__wrapped__(df, "latency_ms", k=1.5)
            df_plot = aggregate_series.__wrapped__(df_clean, "latency_ms", rule="1h") if use_resample else df_clean
        t_prep = time.perf_counter() - t0

        t1 = time.perf_counter()
        if use_resample:
            fig = go.Figure()
            for c in ["mean", "min", "max"]:
                fig.add_trace(go.Scatter(x=df_plot["timestamp"], y=df_plot[c], mode="lines", name=c))
            n_pts = len(df_plot) * 3
        else:
            fig = px.line(df_plot, x="timestamp", y="latency_ms")
            n_pts = len(df_plot)
        _ = fig.to_json()
        t_fig = time.perf_counter() - t1

    return {
        "T_prep": t_prep,
        "T_fig": t_fig,
        "T_srv": t_prep + t_fig,
        "RSS_MB": mon.peak_mb,
        "points": n_pts,
    }


def benchmark_dataset(csv_path: str, n_repeats: int = 10) -> pd.DataFrame:
    modes = [
        ("C0", False, False, "none"),
        ("C1-cold", True, False, "cold"),
        ("C1-warm", True, False, "warm"),
        ("C2-cold", False, True, "none"),
        ("C3-warm", True, True, "warm"),
    ]
    rows = []
    for name, use_cache, use_resample, cache_mode in modes:
        if cache_mode == "warm":
            st.cache_data.clear()
            run_single_pass(csv_path, use_cache=True, use_resample=use_resample)  # прогрів кешу

        runs = []
        for _ in range(n_repeats):
            if cache_mode == "cold":
                st.cache_data.clear()
            runs.append(run_single_pass(csv_path, use_cache=use_cache, use_resample=use_resample))

        res_df = pd.DataFrame(runs)
        rows.append({
            "Mode": name,
            "T_prep_mean": res_df["T_prep"].mean(),
            "T_prep_sd": res_df["T_prep"].std(ddof=1),
            "T_fig_mean": res_df["T_fig"].mean(),
            "T_fig_sd": res_df["T_fig"].std(ddof=1),
            "T_srv_mean": res_df["T_srv"].mean(),
            "T_srv_sd": res_df["T_srv"].std(ddof=1),
            "T_srv_median": res_df["T_srv"].median(),
            "RSS_mean": res_df["RSS_MB"].mean(),
            "RSS_sd": res_df["RSS_MB"].std(ddof=1),
            "Points": int(res_df["points"].iloc[0]),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    csv_file = "data/telemetry_100k.csv"
    generate_telemetry(n_rows=100_000, seed=42).to_csv(csv_file, index=False)
    summary = benchmark_dataset(csv_file, n_repeats=10)
    print(summary.to_string(index=False))
