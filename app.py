# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="PPO Portfolio Dashboard", layout="wide")
st.title("📊 PPO Portfolio Dashboard")

# --- Load data ---
st.sidebar.header("Load evaluation log")
uploaded = st.sidebar.file_uploader("Upload evaluation_log.csv", type=["csv"], accept_multiple_files=False)

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    st.sidebar.info("No file uploaded. Looking for local evaluation_log.csv …")
    try:
        df = pd.read_csv("evaluation_log_3.csv")
    except Exception:
        st.warning("Upload an evaluation_log.csv to continue.")
        st.stop()

# --- Basic cleanup ---
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Step"] = pd.to_numeric(df["Step"], errors="coerce")

# identify tickers from columns like <T>_weight (matches your sample: AAPL_weight, GOOG_weight, MSFT_weight)
tickers = sorted([c[:-7] for c in df.columns if c.endswith("_weight")])

# Ensure numeric types for metrics
for col in ["Reward", "CumulativeReward", "Cash_after", "Turnover", "GrossExposure"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --- Metrics (use equity) ---
equity = df["Cash_after"].to_numpy(dtype=float)
eq_rets = (equity[1:] / np.clip(equity[:-1], 1e-12, None) - 1.0) if len(equity) > 1 else np.array([])

def sharpe_ratio_from_equity(returns: np.ndarray) -> float:
    if returns.size == 0:
        return float("nan")
    mean, std = returns.mean(), returns.std() + 1e-12
    return (mean / std) * np.sqrt(252)

def max_drawdown(series: np.ndarray) -> float:
    if series.size == 0:
        return float("nan")
    roll_max = np.maximum.accumulate(series)
    dd = series / roll_max - 1.0
    return float(dd.min())

def cagr(series: np.ndarray, periods_per_year: int = 252) -> float:
    if series.size < 2:
        return float("nan")
    total_return = series[-1] / np.clip(series[0], 1e-12, None) - 1.0
    years = (series.size - 1) / periods_per_year
    if years <= 0:
        return float("nan")
    return (1.0 + total_return) ** (1.0 / years) - 1.0

sharpe = sharpe_ratio_from_equity(eq_rets)
mdd = max_drawdown(equity)
cagr_val = cagr(equity)
final_cash = equity[-1] if equity.size else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sharpe (equity, ann.)", f"{sharpe:.2f}" if np.isfinite(sharpe) else "—")
c2.metric("Max Drawdown", f"{mdd:.2%}" if np.isfinite(mdd) else "—")
c3.metric("CAGR (≈)", f"{cagr_val:.2%}" if np.isfinite(cagr_val) else "—")
c4.metric("Final Cash", f"${final_cash:,.0f}" if np.isfinite(final_cash) else "—")

# --- X-axis toggle ---
x_axis = st.radio("X-axis", ["Step", "Date"], horizontal=True)
x_col = "Date" if (x_axis == "Date" and "Date" in df.columns and df["Date"].notna().any()) else "Step"

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Equity & Reward", "Daily Returns", "Weights", "Turnover & Exposure", "Table"]
)

with tab1:
    colA, colB = st.columns(2)
    with colA:
        fig = px.line(df, x=x_col, y="Cash_after", title="💰 Portfolio Cash")
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        if "CumulativeReward" in df.columns:
            fig = px.line(df, x=x_col, y="CumulativeReward", title="📈 Cumulative Reward (env)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("CumulativeReward column not found.")

with tab2:
    # Rewards per step + histogram
    if "Reward" in df.columns:
        fig2 = px.line(df, x=x_col, y="Reward", title="Reward (per step)")
        st.plotly_chart(fig2, use_container_width=True)
        fig = px.histogram(df, x="Reward", nbins=50, title="📉 Reward Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Reward column not found.")
    # Per-ticker return distribution if present (e.g., AAPL_return)
    ret_cols = [f"{t}_return" for t in tickers if f"{t}_return" in df.columns]
    if ret_cols:
        melt = df[[x_col] + ret_cols].melt(id_vars=[x_col], var_name="Ticker", value_name="Return")
        melt["Ticker"] = melt["Ticker"].str.replace("_return", "", regex=False)
        figr = px.histogram(melt, x="Return", color="Ticker", barmode="overlay", nbins=60,
                            title="Per-Ticker Return Distribution")
        st.plotly_chart(figr, use_container_width=True)

with tab3:
    if tickers:
        weights_df = df[[x_col] + [f"{t}_weight" for t in tickers]].copy()
        weights_df = weights_df.rename(columns={f"{t}_weight": t for t in tickers})
        fig_w = go.Figure()
        for t in tickers:
            fig_w.add_trace(go.Scatter(
                x=weights_df[x_col], y=weights_df[t], name=t, stackgroup="one", mode="lines"
            ))
        fig_w.update_layout(title="🧮 Portfolio Weights (stacked area)",
                            yaxis=dict(range=[0, 1], title="Weight"))
        st.plotly_chart(fig_w, use_container_width=True)

        # Average & peak weights
        with st.expander("Weight stats"):
            avg_w = weights_df[tickers].mean(numeric_only=True).rename("avg_weight")
            max_w = weights_df[tickers].max(numeric_only=True).rename("max_weight")
            stats = pd.concat([avg_w, max_w], axis=1)
            st.dataframe(stats.style.format("{:.3f}"))
    else:
        st.info("No *_weight columns found to draw weights.")

with tab4:
    # Turnover & Exposure over time
    sub_cols = st.columns(2)
    if "Turnover" in df.columns:
        with sub_cols[0]:
            fig_t = px.line(df, x=x_col, y="Turnover", title="🔁 Turnover per step")
            st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Turnover column not found in the log.")
    if "GrossExposure" in df.columns:
        with sub_cols[1]:
            fig_e = px.line(df, x=x_col, y="GrossExposure", title="📏 Gross Exposure (|weights| sum)")
            st.plotly_chart(fig_e, use_container_width=True)

with tab5:
    # simple filters
    st.write("Use filters then download the filtered table if needed.")
    if df["Step"].notna().any():
        smin, smax = int(df["Step"].min()), int(df["Step"].max())
    else:
        smin, smax = 0, 0
    start, end = st.slider("Step range", smin, smax, (smin, smax))
    view = df[(df["Step"] >= start) & (df["Step"] <= end)]
    st.dataframe(view, use_container_width=True)
    st.download_button(
        "Download filtered CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="evaluation_log_filtered.csv",
        mime="text/csv",
    )

st.caption("Tip: toggle the x-axis to Date to compare across runs, and use Turnover/Exposure to spot DD drivers.")
