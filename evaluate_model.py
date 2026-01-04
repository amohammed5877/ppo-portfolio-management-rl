import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from env import PortfolioEnv

MODEL_PATH = "ppo_portfolio_model_3.zip"
VECNORM_PATH = "vecnormalize_portfolio_env_3.pkl"
TEST_CSV = "stock_test_long.csv"
OUT_LOG = "evaluation_log_3.csv"
OUT_PLOT = "evaluation_plot_3.png"

# --------------------------
# Load & reshape test data
# --------------------------
df = pd.read_csv(TEST_CSV)

# infer tickers from wide columns like "AAPL_Open"
tickers = sorted({c.split("_")[0] for c in df.columns if "_" in c})

long_df = []
for t in tickers:
    cols = [f"{t}_Open", f"{t}_High", f"{t}_Low", f"{t}_Close", f"{t}_Volume"]
    if all(c in df.columns for c in cols):
        tmp = df[["Date"] + cols].copy()
        tmp.columns = ["date", "Open", "High", "Low", "Close", "Volume"]
        tmp["ticker"] = t
        tmp["sentiment"] = 0.0  # If you have test sentiment, merge it instead of 0.0
        # If your data already has lag features per ticker, you can add them here too.
        long_df.append(tmp)

if not long_df:
    raise ValueError("No valid ticker OHLCV columns found in test CSV.")

long_df = pd.concat(long_df, ignore_index=True)
long_df["date"] = pd.to_datetime(long_df["date"])
# Ensure eval has the same features used in training (add sentiment lags)
long_df = long_df.sort_values(["ticker", "date"])
for L in (1, 2, 3):
    col = f"sentiment_lag{L}"
    if col not in long_df.columns:
        long_df[col] = (
            long_df.groupby("ticker")["sentiment"].shift(L).fillna(0.0)
        )

# --------------------------
# Eval env factory
# --------------------------
# IMPORTANT: risk_penalty MUST match your training setting. If you trained with 0.0, keep 0.0 here.
RISK_PENALTY = 0.10  # <<< set this to the SAME value used in train.py

def make_env():
    return PortfolioEnv(
        data=long_df,
        tickers=tickers,
        window=10,
        initial_cash=10_000.0,
        ticker_col="ticker",
        fee=0.0005,
        max_weight=0.40,
        smooth_alpha=0.3,
        risk_penalty=RISK_PENALTY,
    )

eval_env = DummyVecEnv([make_env])

# If you trained with VecNormalize, load its stats for evaluation
if Path(VECNORM_PATH).exists():
    eval_env = VecNormalize.load(VECNORM_PATH, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False  # don't normalize rewards at eval
else:
    print("⚠️ VecNormalize stats not found; evaluating without normalization.")

# --------------------------
# Load the trained model
# --------------------------
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = PPO.load(MODEL_PATH, env=eval_env)

# --------------------------
# Rollout
# --------------------------
obs = eval_env.reset()
done = np.array([False])
total_reward = 0.0
step = 0
log = []

# Access the underlying raw env for logging internals
raw_env = eval_env.envs[0]
N, T = raw_env.N, raw_env.T

print(f"\n🔍 Starting Evaluation for tickers ({len(tickers)}): {tickers}\n")

while True:
    # predict deterministic action
    action, _ = model.predict(obs, deterministic=True)

    # Pre-step logging (date, prev close, etc.)
    t = raw_env.step_idx  # current time index before step advances
    date = raw_env.dates[t] if t < len(raw_env.dates) else None

    # Compute per-asset returns from close-to-close (t-1 -> t)
    if 1 <= t < T:
        prev_close = raw_env.closes[t - 1, :]
        curr_close = raw_env.closes[t, :]
        rets = (curr_close - prev_close) / np.clip(prev_close, 1e-12, None)
    else:
        rets = np.zeros(N, dtype=np.float32)

    # ===== NEW: capture prev_weights BEFORE stepping =====
    prev_weights = raw_env.weights.copy()

    # Step the env
    obs, reward, done, _ = eval_env.step(action)
    total_reward += float(reward[0])

    # After step, the env has updated cash, weights, step_idx, etc.
    post_weights = raw_env.weights.copy()

    # Safe closes at new t (after step the env usually moves to next index)
    t_after = raw_env.step_idx
    if t_after < T:
        closes_after = raw_env.closes[t_after - 1, :] if t_after > 0 else raw_env.closes[0, :]
    else:
        closes_after = np.full(N, np.nan, dtype=np.float32)

    # Build a row for CSV
    row = {
        "Step": step,
        "Date": pd.to_datetime(str(date)),
        "Reward": float(reward[0]),
        "CumulativeReward": float(total_reward),
        "Cash_after": float(raw_env.cash),
    }

    # Per-ticker diagnostics
    for i, tk in enumerate(tickers):
        row[f"{tk}_return"] = float(rets[i])
        row[f"{tk}_weight"] = float(post_weights[i])
        row[f"{tk}_close"] = float(closes_after[i]) if np.isfinite(closes_after[i]) else np.nan

    # ===== NEW: turnover & exposure (outside the per-ticker loop) =====
    row["Turnover"] = float(np.sum(np.abs(post_weights - prev_weights)))
    row["GrossExposure"] = float(np.sum(np.abs(post_weights)))

    log.append(row)

    step += 1
    if done[0] or raw_env.step_idx >= T - 1:
        break
print(f"\n✅ Evaluation finished in {step} steps. Final cumulative reward: {total_reward:.6f}")

# --------------------------
# Save log
# --------------------------
log_df = pd.DataFrame(log)
log_df.to_csv(OUT_LOG, index=False)
print(f"📁 Saved evaluation log to {OUT_LOG}")

# --------------------------
# Equity-based metrics
# --------------------------
equity = log_df["Cash_after"].to_numpy()
if len(equity) < 2:
    raise RuntimeError("Not enough steps to compute metrics.")

eq_rets = equity[1:] / np.clip(equity[:-1], 1e-12, None) - 1.0

mean_r = float(np.nanmean(eq_rets))
vol_r = float(np.nanstd(eq_rets) + 1e-12)
sharpe = (mean_r / vol_r) * np.sqrt(252.0)

roll_max = np.maximum.accumulate(equity)
mdd = float(np.min(equity / roll_max - 1.0))  # negative number

# CAGR (assume 252 trading days per year)
period_years = max(len(eq_rets) / 252.0, 1e-9)
cagr = float((equity[-1] / np.clip(equity[0], 1e-12, None)) ** (1.0 / period_years) - 1.0)

print(f"📈 Sharpe (equity): {sharpe:.3f} | Vol (daily): {vol_r:.4f} | Max DD: {mdd:.2%} | "
      f"CAGR: {cagr:.2%} | Final Cash: ${equity[-1]:,.2f}")

# --------------------------
# Plots
# --------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(log_df["CumulativeReward"].values)
plt.title("Cumulative Reward (env)")
plt.xlabel("Step")
plt.ylabel("Reward")

plt.subplot(1, 2, 2)
plt.plot(log_df["Cash_after"].values)
plt.title("Portfolio Cash (equity)")
plt.xlabel("Step")
plt.ylabel("Cash ($)")

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150)
print(f"🖼️ Plot saved as {OUT_PLOT}")

# Show plot in interactive runs
try:
    plt.show()
except Exception:
    pass
