import time
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pipeline import aggregate_series, clean_data, load_data
from benchmark import generate_telemetry

st.set_page_config(page_title="Телеметрія мережі — Аналітична система", layout="wide")
st.title("Інтерактивна візуалізація телеметрії телекомунікаційних мереж")

with st.sidebar:
    st.header("Параметри конвеєра")
    source_mode = st.radio("Джерело даних", ["Синтетичний генератор", "CSV-файл"])
    metric = st.selectbox("Показник", ["latency_ms", "throughput_mbps", "packet_loss_pct"])
    k_iqr = st.slider("Коефіцієнт IQR (k)", 1.0, 3.0, 1.5, 0.1)
    use_resampling = st.checkbox("Погодинна агрегація (1h)", value=True)
    n_rows = st.selectbox("Обсяг синтетичної вибірки", [10_000, 100_000, 500_000, 1_000_000], index=1)

t0 = time.perf_counter()
if source_mode == "Синтетичний генератор":
    raw_df = generate_telemetry(n_rows=n_rows, seed=42)
else:
    uploaded = st.file_uploader("Завантажити CSV (timestamp, latency_ms, ...)", type=["csv"])
    if uploaded is None:
        st.info("Завантажте CSV-файл або оберіть синтетичний генератор.")
        st.stop()
    raw_df = load_data(uploaded)

cleaned_df, q_report = clean_data(raw_df, col=metric, k=k_iqr)
plot_df = aggregate_series(cleaned_df, col=metric, rule="1h") if use_resampling else cleaned_df
t_prep = time.perf_counter() - t0

t1 = time.perf_counter()
if use_resampling:
    fig = go.Figure()
    for stat_col in ["mean", "min", "max"]:
        fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df[stat_col], mode="lines", name=stat_col))
    fig.update_layout(title=f"Агрегований ряд ({metric}, 1h)", xaxis_title="Час (UTC)", yaxis_title=metric)
    n_pts = len(plot_df) * 3
else:
    fig = px.line(plot_df, x="timestamp", y=metric, title=f"Повний очищений ряд ({metric})")
    n_pts = len(plot_df)
_ = fig.to_json()
t_fig = time.perf_counter() - t1

c1, c2, c3, c4 = st.columns(4)
c1.metric("T_prep, с", f"{t_prep:.3f}")
c2.metric("T_fig, с", f"{t_fig:.3f}")
c3.metric("T_srv, с", f"{t_prep + t_fig:.3f}")
c4.metric("Точок у фігурі", f"{n_pts:,}".replace(",", " "))

st.plotly_chart(fig, use_container_width=True)
with st.expander("Звіт якості даних (Шар 2)"):
    st.json(q_report)
