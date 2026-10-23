"""Multi-TF panel + teacher situation card (eye/brain v2 step 1, 2026-10-23).

v1 gave the teacher ONE 72-bar H1 strip — a champion never reads a single timeframe.
This renders the champion's actual scan: D1 -> H4 -> H1 -> M15 price panes (same template:
BB solid, Keltner dashed, Alligator shifted, Ichimoku cloud ahead) + one H1 sub-pane row
(OBV/AO/AC), then asks the teacher for a SITUATION CARD (structured JSON), read top-down —
regime, structure levels in ATR units, stretch, vol state, clarity, invalidation.

Runs on the VPS (mt5 attach + vision_client key). No dates/ids on the image.

    python panel_situation_card.py            # render panel + one card
    python panel_situation_card.py --render-only panel.png
"""
import argparse
import base64
import json
import re
import sys
from pathlib import Path

HERE = Path(r"C:\Users\Administrator\vision_regime") if r"C:\Users\Administrator" in __file__ or Path(r"C:\Users\Administrator").exists() else Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, r"C:\Users\Administrator")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from render_jay_template_20261022 import panes, BG, FG, GRID  # noqa: E402

SHOW, AHEAD = 72, 26
TFS = ("D1", "H4", "H1", "M15")
TF_MT5 = {"M15": "TIMEFRAME_M15", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4", "D1": "TIMEFRAME_D1"}

CARD_PROMPT = (
    "You are a discretionary gold trader scanning an XAUUSD multi-timeframe panel, TOP to BOTTOM: "
    "D1, H4, H1, M15 (72 candles each, last candle on the right is now). Each pane: candles, Bollinger "
    "bands (solid grey), Keltner channel (dashed violet), Alligator jaw/teeth/lips (blue/red/green, shifted "
    "forward), Ichimoku cloud (shaded, drawn ahead). Bottom row: OBV, AO, AC for H1. Read the panel "
    "TOP-DOWN like a champion: higher timeframe sets the regime, lower timeframes show the execution state. "
    "Reply with ONLY one JSON object: "
    '{"regime": "trend_up|trend_down|range|transition", '
    '"vol_state": "contracting|normal|expanding", '
    '"trend_clarity": 0-3, '
    '"stretch": "overextended_up|overextended_down|neutral", '
    '"structure": {"nearest_support_atr": <float, distance below last price in ATR units, negative if none visible>, '
    '"nearest_resistance_atr": <float, distance above last price in ATR units, negative if none visible>}, '
    '"invalidation": "<one line: what price action would prove this read wrong>", '
    '"notes": "<one line max>"}. '
    "Be concrete and terse; estimate ATR from the H1 pane's recent candle sizes.")


def price_strip(ax, h, title):
    p = panes(h)
    n = len(h)
    s = slice(n - SHOW, n)
    x = np.arange(SHOW)
    o, hi, lo, c = (h[k].to_numpy()[s] for k in ("open", "high", "low", "close"))
    col = np.where(c >= o, "#26b496", "#f05652")
    ax.vlines(x, lo, hi, color=col, lw=1)
    ax.bar(x, np.maximum(np.abs(c - o), 1e-6), bottom=np.minimum(o, c), width=0.7, color=col)
    ax.plot(x, p["bb_up"].to_numpy()[s], color="#8fa3c0", lw=0.9)
    ax.plot(x, p["bb_dn"].to_numpy()[s], color="#8fa3c0", lw=0.9)
    ax.plot(x, p["kc_up"].to_numpy()[s], color="#c08fbf", lw=0.9, ls="--")
    ax.plot(x, p["kc_dn"].to_numpy()[s], color="#c08fbf", lw=0.9, ls="--")
    for k, shift, cl in (("jaw", 8, "#3d7fe0"), ("teeth", 5, "#e04b4b"), ("lips", 3, "#3fc46a")):
        ax.plot(np.arange(SHOW + shift), p[k].to_numpy()[n - SHOW - shift:n], color=cl, lw=1.2)
    xa = np.arange(SHOW + AHEAD)
    cs = slice(n - SHOW - AHEAD, n)
    sa, sb = p["span_a"].to_numpy()[cs], p["span_b"].to_numpy()[cs]
    ax.fill_between(xa, sa, sb, where=sa >= sb, color="#2e8b57", alpha=0.18, lw=0)
    ax.fill_between(xa, sa, sb, where=sa < sb, color="#b22222", alpha=0.18, lw=0)
    ax.set_xlim(-1, SHOW + AHEAD)
    ax.text(0.005, 0.9, title, transform=ax.transAxes, color=FG, fontsize=10, va="top")
    ax.yaxis.tick_right()


def render_panel(frames: dict, path) -> None:
    fig = plt.figure(figsize=(11, 15), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(7, 1, height_ratios=[4, 4, 4, 4, 1, 1, 1], hspace=0.08)
    axes = []
    for i, tf in enumerate(TFS):
        ax = fig.add_subplot(gs[i])
        price_strip(ax, frames[tf], f"XAUUSD {tf}")
        axes.append(ax)
    from render_jay_template_20261022 import bars_hist
    h1 = frames["H1"]
    p1 = panes(h1)
    n = len(h1)
    s = slice(n - SHOW, n)
    x = np.arange(SHOW)
    for j, (k, kind) in enumerate((("obv", "line"), ("ao", "hist"), ("ac", "hist")), start=4):
        a = fig.add_subplot(gs[j], sharex=axes[0])
        v = p1[k].to_numpy()[s]
        if kind == "line":
            a.plot(x, v, color="#e0b85a", lw=1.0)
        else:
            bars_hist(a, x, v)
        a.text(0.005, 0.75, k.upper() + " (H1)", transform=a.transAxes, color=FG, fontsize=8)
        a.set_yticks([])
        axes.append(a)
    for a in axes:
        a.set_facecolor(BG)
        a.grid(color=GRID, lw=0.5)
        a.tick_params(colors=FG, labelsize=7, labelbottom=False, bottom=False)
        for sp in a.spines.values():
            sp.set_color(GRID)
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def parse_card(text):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except ValueError:
        return None
    if j.get("regime") not in ("trend_up", "trend_down", "range", "transition"):
        return None
    if j.get("vol_state") not in ("contracting", "normal", "expanding"):
        return None
    if j.get("trend_clarity") not in (0, 1, 2, 3):
        return None
    if j.get("stretch") not in ("overextended_up", "overextended_down", "neutral"):
        return None
    if not isinstance(j.get("structure"), dict) or "invalidation" not in j:
        return None
    return j


def card_label(key, png, model="openai/gpt-6.1-sol"):
    import time
    import urllib.request
    import vision_client as V
    payload = {"model": model, "max_tokens": 1200, "temperature": 0.2,
               "messages": [{"role": "user", "content": [
                   {"type": "text", "text": CARD_PROMPT},
                   {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                                       base64.b64encode(png.read_bytes()).decode()}}]}]}
    req = urllib.request.Request(V.OR_URL, data=json.dumps(payload).encode(),
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json",
                                          "X-Title": "vision_situation_card"})
    t0 = time.time()
    try:
        body = json.load(urllib.request.urlopen(req, timeout=120))
    except Exception as e:
        return {"error": type(e).__name__, "cost": 0.0, "sec": round(time.time() - t0, 1)}
    ch = (body.get("choices") or [{}])[0]
    text = V._extract(ch.get("message") or {})
    return {"card": parse_card(text), "raw": text[:800],
            "cost": float((body.get("usage") or {}).get("cost") or 0.0),
            "sec": round(time.time() - t0, 1)}


def pull_frames():
    import MetaTrader5 as mt5
    import pandas as pd
    assert mt5.initialize(), mt5.last_error()  # bare attach
    frames = {}
    for tf in TFS:
        r = mt5.copy_rates_from_pos("XAUUSD", getattr(mt5, TF_MT5[tf]), 0, 260)
        if r is None or len(r) < 240:
            mt5.shutdown()
            raise RuntimeError(f"not enough {tf} bars")
        frames[tf] = pd.DataFrame({"open": r["open"], "high": r["high"], "low": r["low"],
                                   "close": r["close"], "tick_volume": r["tick_volume"]})
    last = int(mt5.symbol_info_tick("XAUUSD").time) if mt5.symbol_info_tick("XAUUSD") else 0
    mt5.shutdown()
    return frames, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-only")
    a = ap.parse_args()
    frames, _ = pull_frames()
    out = Path(a.render_only) if a.render_only else HERE / "_panel.png"
    render_panel(frames, out)
    print(f"panel rendered: {out}", flush=True)
    if a.render_only:
        return
    import vision_client as V
    res = card_label(V._load_env(), out)
    print(json.dumps(res.get("card") or res, indent=2)[:900])


if __name__ == "__main__":
    main()
