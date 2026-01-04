import gym
import numpy as np
import pandas as pd
from gym.utils import seeding

FEATURES = ["Open", "High", "Low", "Close", "Volume", "sentiment"]

class PortfolioEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        tickers,
        window: int = 10,
        initial_cash: float = 10_000.0,
        ticker_col: str = "ticker",
        fee: float = 0.0,
        max_weight: float = 0.40,       # cap per-asset weight
        smooth_alpha: float = 0.3,     # action smoothing (0..1)
        risk_penalty: float = 0.10      # drawdown penalty strength
    ):
        """
        data: long dataframe with columns: [date, <ticker_col>, Open, High, Low, Close, Volume, sentiment]
        """
        super().__init__()
        self.window = int(window)
        self.initial_cash = float(initial_cash)
        self.tickers = list(tickers)
        self.ticker_col = ticker_col
        self.fee = float(fee)
        self.max_weight = float(max_weight)
        self.smooth_alpha = float(smooth_alpha)
        self.risk_penalty = float(risk_penalty)

        df = data.copy()
        # Auto-include sentiment_lag* columns if present
        lag_cols = [c for c in df.columns if c.startswith("sentiment_lag")]
        for c in lag_cols:
            if c not in FEATURES:
                FEATURES.append(c)

        # Normalize ticker column name
        if self.ticker_col not in df.columns:
            if "Ticker" in df.columns:
                df = df.rename(columns={"Ticker": self.ticker_col})
            elif "symbol" in df.columns:
                df = df.rename(columns={"symbol": self.ticker_col})
            else:
                raise ValueError(f"Ticker column '{self.ticker_col}' not found in data.")

        # Ensure date column exists and is datetime
        date_col = None
        for cand in ["date", "Date", "datetime", "Datetime", "timestamp"]:
            if cand in df.columns:
                date_col = cand
                break
        if date_col is None:
            if df.index.name is not None:
                try:
                    df["date"] = pd.to_datetime(df.index)
                    date_col = "date"
                except Exception:
                    raise ValueError("No usable date column/index found.")
            else:
                raise ValueError("No usable date column found.")
        df[date_col] = pd.to_datetime(df[date_col])

        # Columns check
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Data is missing columns: {missing}")

        # Build per-ticker aligned arrays
        per_ticker = {}
        for t in self.tickers:
            dft = (
                df[df[self.ticker_col] == t]
                .sort_values(date_col)
                .set_index(date_col)[FEATURES]
                .dropna()
            )
            per_ticker[t] = dft

        # Intersect dates so all tickers share same timeline
        common_idx = None
        for _, dft in per_ticker.items():
            common_idx = dft.index if common_idx is None else common_idx.intersection(dft.index)
        if common_idx is None or len(common_idx) <= self.window:
            raise ValueError("Not enough overlapping data across tickers for the given window.")

        panels, closes = [], []
        for t in self.tickers:
            d = per_ticker[t].loc[common_idx]
            panels.append(d.values.astype(np.float32))          # (T, F)
            closes.append(d["Close"].values.astype(np.float32)) # (T,)
        self.panel = np.stack(panels, axis=1)                   # (T, N, F)
        self.closes = np.stack(closes, axis=1)                  # (T, N)
        self.dates = np.array(common_idx)
        self.T, self.N, self.F = self.panel.shape

        # Spaces
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window, self.N, self.F), dtype=np.float32
        )

        # Internal state
        self._np_random = None
        self.seed()
        self.reset()

    def seed(self, seed: int = None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation(self):
        obs = self.panel[self.step_idx - self.window:self.step_idx, :, :]
        return np.nan_to_num(obs, copy=False)

    def reset(self):
        self.step_idx = self.window
        self.cash = self.initial_cash
        self.peak_cash = self.cash       # for drawdown
        self.prev_drawdown = 0.0
        self.weights = np.full(self.N, 1.0 / self.N, dtype=np.float32)
        return self._get_observation()

    def step(self, actions):
        # ----- 1) Build next target weights (cap + smooth + renorm) -----
        # previous weights snapshot (used for turnover calc)
        prev_weights = self.weights.copy()

        # normalize raw action to simplex
        a = np.nan_to_num(actions).astype(np.float32)
        s = float(a.sum())
        a = prev_weights if (s <= 0.0 or not np.isfinite(s)) else (a / s)

        # cap per-asset weight and renormalize
        if self.max_weight < 1.0:
            a = np.clip(a, 0.0, self.max_weight)
            s = float(a.sum())
            a = (a / s) if s > 1e-9 else prev_weights

        # smooth actions (EMA towards new target), then renormalize
        if self.smooth_alpha > 0.0:
            a = self.smooth_alpha * a + (1.0 - self.smooth_alpha) * prev_weights
            s = float(a.sum())
            a = (a / s) if s > 1e-9 else prev_weights

        # ----- 2) End episode check / build returns -----
        t = self.step_idx
        if t >= self.T:
            return self._get_observation(), 0.0, True, {}

        # close-to-close returns (t-1 -> t)
        prev_prices = self.closes[t - 1, :]
        curr_prices = self.closes[t, :]
        rets = (curr_prices - prev_prices) / np.clip(prev_prices, 1e-12, None)

        # ----- 3) Portfolio return and costs -----
        # gross weighted return
        gross_ret = float(np.dot(a, rets))

        # turnover based on change in weights (post-cap/smoothing)
        turnover = float(np.abs(a - prev_weights).sum())

        # explicit transaction cost (proportional to turnover)
        cost = self.fee * turnover if self.fee > 0.0 else 0.0

        # optional behavioral turnover penalty (independent of market fee)
        turnover_penalty = 0.0025 * turnover  # tune 0.001–0.01

        # net portfolio return after explicit costs (as a fraction of equity)
        portfolio_return = gross_ret - cost

        # next cash (apply net return)
        cash_next = self.cash * (1.0 + portfolio_return)

        # ----- 4) Drawdown (incremental) penalty -----
        prev_peak = self.peak_cash
        new_peak = max(prev_peak, cash_next)

        prev_dd = 1.0 - (self.cash / max(prev_peak, self.cash))     # previous drawdown
        new_dd  = 1.0 - (cash_next / new_peak)                      # new drawdown
        dd_increase = max(new_dd - prev_dd, 0.0)                    # penalize only worsening DD

        dd_penalty = self.risk_penalty * dd_increase

        # total reward signal (unclipped)
        reward = float(np.nan_to_num(portfolio_return - dd_penalty - turnover_penalty))

        # ----- 5) Update state -----
        self.weights = a
        self.cash = cash_next
        self.peak_cash = new_peak
        self.prev_drawdown = new_dd
        self.step_idx += 1

        done = self.step_idx >= self.T
        obs = self._get_observation()
        info = {
            "portfolio_return": portfolio_return,
            "turnover": turnover,
            "tx_cost": cost,
            "turnover_penalty": turnover_penalty,
            "dd": float(new_dd),
            "dd_penalty": dd_penalty,
        }
        return obs, reward, done, info

    @property
    def current_date(self):
        return self.dates[self.step_idx] if 0 <= self.step_idx < len(self.dates) else None
