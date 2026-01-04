# sentiment.py
from __future__ import annotations
import pandas as pd
from typing import Iterable, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pytz

_ANALYZER = SentimentIntensityAnalyzer()
_NY_TZ = pytz.timezone("America/New_York")


def _to_trading_date(ts: pd.Series) -> pd.Series:
    """Convert timestamps to US trading 'date' in America/New_York."""
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    # if series is naive (no tz), treat as UTC first
    dt = dt.dt.tz_convert(_NY_TZ).dt.date
    return dt


def analyze_sentiment(news_df: pd.DataFrame,
                      text_col: str = "headline",
                      out_col: str = "sentiment") -> pd.DataFrame:
    """Add VADER compound score column to a copy of news_df."""
    df = news_df.copy()
    if text_col not in df.columns:
        raise ValueError(f"Expected column '{text_col}' in news_df.")
    df[out_col] = df[text_col].astype(str).apply(lambda x: _ANALYZER.polarity_scores(x)["compound"])
    return df


def merge_sentiment_with_prices(
    price_df: pd.DataFrame,
    news_df: pd.DataFrame,
    price_ticker_col: str = "Ticker",   # or "symbol" in your price data
    price_date_col_hint: Optional[str] = None,  # if you know your date col, pass it
    news_symbol_col: str = "symbol",
    news_datetime_col: str = "datetime",
    news_headline_col: str = "headline",
    add_lags: Iterable[int] = (1, 2, 3),
) -> pd.DataFrame:
    """
    Merge per-(date, ticker) sentiment into price_df.
    Expects:
      - price_df has ['<date>', price_ticker_col, ...]
      - news_df has [news_symbol_col, news_datetime_col, news_headline_col]
    Returns price_df with columns:
      - 'sentiment' and optional 'sentiment_lag{n}'
    """

    # ---------- 1) Normalize price_df columns ----------
    p = price_df.copy()

    # Flatten MultiIndex columns if needed
    if isinstance(p.columns, pd.MultiIndex):
        p.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in p.columns]

    # Ensure ticker column exists
    if price_ticker_col not in p.columns:
        # try common alternatives
        for alt in ["symbol", "Symbol", "ticker", "TICKER"]:
            if alt in p.columns:
                p = p.rename(columns={alt: price_ticker_col})
                break
        else:
            raise ValueError(f"Ticker column '{price_ticker_col}' not found in price_df.")

    # Ensure a date column exists
    if price_date_col_hint and price_date_col_hint in p.columns:
        p["date"] = pd.to_datetime(p[price_date_col_hint]).dt.date
    else:
        # try common names or index
        for cand in ["date", "Date", "DATE", "timestamp", "Datetime", "datetime"]:
            if cand in p.columns:
                p["date"] = pd.to_datetime(p[cand]).dt.date
                break
        else:
            # fall back to index if it looks datetime-like
            if p.index.name:
                try:
                    p["date"] = pd.to_datetime(p.index).date
                except Exception:
                    raise ValueError("No usable date column found in price_df.")
            else:
                raise ValueError("No usable date column found in price_df.")

    # ---------- 2) Build per-(date,ticker) sentiment ----------
    n = news_df.copy()

    # If the CSV already has 'datetime' and 'symbol' (your case)
    # create trading 'date' and a common ticker column matching price_df
    if news_datetime_col not in n.columns:
        # fallback: maybe they already have 'date'
        if "date" not in n.columns:
            raise ValueError(f"'{news_datetime_col}' or 'date' must exist in news_df.")
        n["date"] = pd.to_datetime(n["date"]).dt.date
    else:
        n["date"] = _to_trading_date(n[news_datetime_col])

    if news_symbol_col not in n.columns:
        # fallback: maybe they already have 'Ticker'
        if "Ticker" in n.columns:
            n = n.rename(columns={"Ticker": news_symbol_col})
        else:
            raise ValueError(f"'{news_symbol_col}' not found in news_df.")

    # Sentiment score
    if "sentiment" not in n.columns:
        n = analyze_sentiment(n, text_col=news_headline_col, out_col="sentiment")

    # Keep only needed columns
    n = n[[news_symbol_col, "date", "sentiment"]].dropna(subset=["date"])

    # Aggregate by (date, symbol) — mean is a good simple reducer
    n_daily = (
        n.groupby(["date", news_symbol_col])["sentiment"]
         .mean()
         .reset_index()
         .rename(columns={news_symbol_col: price_ticker_col})
    )

    # ---------- 3) Merge into prices ----------
    merged = p.merge(n_daily, on=["date", price_ticker_col], how="left")
    merged["sentiment"] = merged["sentiment"].fillna(0.0)

    # ---------- 4) Optional: add lagged sentiment features ----------
    if add_lags:
        merged = merged.sort_values(["date", price_ticker_col])
        for L in add_lags:
            merged[f"sentiment_lag{L}"] = (
                merged.groupby(price_ticker_col)["sentiment"].shift(L).fillna(0.0)
            )

    return merged
