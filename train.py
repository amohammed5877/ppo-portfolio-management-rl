from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import PortfolioEnv
from data_loader import load_flat_stock_data
from sentiment import merge_sentiment_with_prices
import pandas as pd

SEED = 42
tickers = ["AAPL", "MSFT", "GOOG"]

# 1) Load prices
price_df = load_flat_stock_data("stock_train_long.csv", tickers)
if "Ticker" not in price_df.columns:
    price_df = price_df.rename(columns={"symbol": "Ticker", "Symbol": "Ticker"})

# 2) Load news, merge per (date, ticker)
news_df = pd.read_csv("stock_news_2025_news.csv")  # columns: symbol, datetime, headline, ...
df = merge_sentiment_with_prices(
    price_df,
    news_df,
    price_ticker_col="Ticker",
    news_symbol_col="symbol",
    news_datetime_col="datetime",
    news_headline_col="headline",
    add_lags=(1, 2, 3),
)

# 3) Env + normalization
def make_env():
    env = PortfolioEnv(
        data=df,       # whatever your merged/priced df variable is
        tickers=tickers,
        window=10,
        initial_cash=10_000.0,
        ticker_col="ticker",
        fee=0.0005,
        max_weight=0.40,
        smooth_alpha=0.3,
        risk_penalty=0.0,                   # <—— use 0.0 in BOTH train & eval (or 0.2 in BOTH)
    )
    env.seed(SEED)                          # this previously did not run due to an early return
    return env


venv = DummyVecEnv([make_env])
venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

# 4) Train PPO  — simple, safer preset
policy_kwargs = dict(net_arch=[128, 128])
model = PPO(
    "MlpPolicy",
    venv,
    seed=SEED,
    n_steps=2048,
    batch_size=256,
    learning_rate=3e-4,  # keep, or try linear schedule -> 3e-4 → 1e-4 over training
    clip_range=0.2,   # was 0.1 (can help stability under reward rescaling)
    n_epochs=5,
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.003,   # was 0.005 (slightly less randomness after decent policy emerges)
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=0.01,
    policy_kwargs=policy_kwargs,
)

model.learn(total_timesteps=500_000)
model.save("ppo_portfolio_model_3")
venv.save("vecnormalize_portfolio_env_3.pkl")  # save normalization stats
