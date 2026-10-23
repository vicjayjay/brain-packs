"""Tape-feel (盘感) indicator bus — the full set Jay actually trades with.

All functions take pandas Series/DataFrames indexed by bar-open time and return
per-bar values that are CAUSAL: the value at bar t uses only data from bars
<= t (Ichimoku spans use the shift-corrected, displayed-now values). No
lookahead anywhere; a truncation test enforces this contract.

Set: OBV + bend, Bill Williams (Alligator jaw/teeth/lips, fractals, AO, AC),
Keltner channel position, Bollinger %B/z, Ichimoku (tenkan, kijun, senkou A/B
as displayed now, TK cross, price vs cloud, chikou), EMA20/90/200 distances +
stack, RSI, bar body/ATR + velocity, tick-volume ratio.

Used by the exit lock/trail study family and the vision-pattern arm (numeric
side — the teacher sees the same indicators rendered on the chart).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff().fillna(0)) * df["tick_volume"].astype(float)).cumsum()


def obv_bend_up(df: pd.DataFrame, lb: int = 5) -> pd.Series:
    """OBV slope sign flips to up (Bill-Williams-style momentum turn)."""
    slope = obv(df).diff(lb)
    bend = (np.sign(slope) != np.sign(slope.shift(lb))) & slope.shift(lb).notna()
    return (bend & (slope > 0)).astype(float)


# ── Bill Williams ────────────────────────────────────────────────────────────

def smma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def alligator(df: pd.DataFrame) -> pd.DataFrame:
    """Jaw (13, shift 8), teeth (8, shift 5), lips (5, shift 3) on the median
    price. Displayed lines are the SMMAs shifted FORWARD; the causal value at t
    of the displayed line is smma.shift(k) at t — i.e. computed from bars <= t-k.
    For feature use we take the RAW smma at t (the line's current level), plus
    mouth openness and alignment."""
    med = (df["high"] + df["low"]) / 2
    jaw = smma(med, 13)
    teeth = smma(med, 8)
    lips = smma(med, 5)
    out = pd.DataFrame(index=df.index)
    out["gator_lips"] = lips
    out["gator_teeth"] = teeth
    out["gator_jaw"] = jaw
    out["gator_mouth_atr"] = (lips - jaw) / atr(df)          # mouth opening, signed
    out["gator_aligned"] = (np.sign(lips - teeth) + np.sign(teeth - jaw)) / 2  # +1 bull stack, -1 bear
    return out


def bw_fractals(df: pd.DataFrame) -> pd.Series:
    """Bill Williams 5-bar fractal, CONFIRMED 2 bars later (causal): +1 up-fractal
    confirmed at t, -1 down-fractal, 0 none."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    out = np.zeros(len(df))
    for i in range(4, len(df)):
        c = i - 2  # candidate bar, confirmed at i
        if h[c] >= max(h[c - 2], h[c - 1], h[c + 1], h[c + 2]):
            out[i] = 1.0
        elif l[c] <= min(l[c - 2], l[c - 1], l[c + 1], l[c + 2]):
            out[i] = -1.0
    return pd.Series(out, index=df.index)


def ao(df: pd.DataFrame) -> pd.Series:
    med = (df["high"] + df["low"]) / 2
    return med.rolling(5).mean() - med.rolling(34).mean()


def ac(df: pd.DataFrame) -> pd.Series:
    a = ao(df)
    return a - a.rolling(5).mean()


# ── Keltner / Bollinger ──────────────────────────────────────────────────────

def keltner_pct(df: pd.DataFrame, n: int = 20, mult: float = 2.0) -> pd.Series:
    """Position within the Keltner channel: 0 = lower rail, 1 = upper rail,
    >1 = above ( Jay's 'touching the ceiling' is pct >= 1)."""
    mid = df["close"].ewm(span=n, adjust=False).mean()
    rng = mult * atr(df, 10)
    return (df["close"] - (mid - rng)) / (2 * rng)


def bollinger(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    sma = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std(ddof=0).replace(0, np.nan)
    return pd.DataFrame({"pctb": (df["close"] - (sma - 2 * sd)) / (4 * sd),
                         "boll_z": (df["close"] - sma) / sd}, index=df.index)


# ── Ichimoku ─────────────────────────────────────────────────────────────────

def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Causal Ichimoku state. tenkan (9), kijun (26), and the cloud AS DISPLAYED
    NOW: senkou A/B shown at t were computed 26 bars ago, so the causal value is
    the shifted-forward span — span.shift(26) is what you SEE at t."""
    h9 = df["high"].rolling(9).max(); l9 = df["low"].rolling(9).min()
    h26 = df["high"].rolling(26).max(); l26 = df["low"].rolling(26).min()
    h52 = df["high"].rolling(52).max(); l52 = df["low"].rolling(52).min()
    tenkan = (h9 + l9) / 2
    kijun = (h26 + l26) / 2
    senkou_a_now = ((tenkan + kijun) / 2).shift(26)
    senkou_b_now = ((h52 + l52) / 2).shift(26)
    out = pd.DataFrame(index=df.index)
    out["ichi_tk_cross"] = np.sign(tenkan - kijun)
    a = atr(df)
    out["ichi_vs_kijun_atr"] = (df["close"] - kijun) / a
    out["ichi_above_cloud"] = (df["close"] > pd.concat([senkou_a_now, senkou_b_now], axis=1).max(axis=1)).astype(float)
    out["ichi_below_cloud"] = (df["close"] < pd.concat([senkou_a_now, senkou_b_now], axis=1).min(axis=1)).astype(float)
    out["ichi_cloud_bull"] = (senkou_a_now > senkou_b_now).astype(float)
    out["ichi_chikou_above"] = (df["close"] > df["close"].shift(26)).astype(float)
    return out


# ── composite bus ────────────────────────────────────────────────────────────

def tape_feel(df: pd.DataFrame) -> pd.DataFrame:
    """The full per-bar 盘感 frame. df: OHLC + tick_volume indexed by bar-open."""
    out = pd.DataFrame(index=df.index)
    c = df["close"].astype(float)
    a14 = atr(df, 14)
    body = c - df["open"].astype(float)
    out["body_atr"] = body / a14
    out["body_vel"] = out["body_atr"] / out["body_atr"].abs().rolling(5).median().replace(0, np.nan)
    out["rsi"] = rsi(c)
    for n in (20, 90, 200):
        out[f"ema{n}_d"] = (c - c.ewm(span=n, adjust=False).mean()) / a14
    e20 = c.ewm(span=20, adjust=False).mean(); e90 = c.ewm(span=90, adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    out["ema_stack"] = np.sign(c - e20) + np.sign(e20 - e90) + np.sign(e90 - e200)
    tv = df["tick_volume"].astype(float)
    out["tvol_ratio"] = tv / tv.shift(1).rolling(1440).mean()
    out["obv_bend_up"] = obv_bend_up(df)
    out["ao"] = ao(df) / a14
    out["ac"] = ac(df) / a14
    out["bw_fractal"] = bw_fractals(df)
    out["keltner_pct"] = keltner_pct(df)
    out = pd.concat([out, alligator(df)[["gator_mouth_atr", "gator_aligned"]],
                     bollinger(df), ichimoku(df)], axis=1)
    return out
