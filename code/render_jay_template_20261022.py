"""Render the vision-pattern arm's charts on Jay's template (kimi's design, vision_pattern_arm_design_20261022.md).

Each manifest row -> a 72-bar H1 chart ending at the anchor (bars strictly before anchor_tape; the last hour is the
partial hour a trader would see). Price pane: candles, Bollinger 20/2, Keltner EMA20 +/- 2 x ATR10, Alligator
jaw/teeth/lips drawn shifted forward 8/5/3 as MT5 shows them, Ichimoku cloud drawn 26 bars ahead. Sub-panes: OBV,
AO, AC (colour = rising/falling, Bill Williams convention). No dates, no ids, no side on the image: the teacher must
not be able to tell a champion moment from its matched random.

    python render_jay_template_20261022.py [--limit N] [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parents[2] if len(HERE.parents) > 2 else None
if ROOT is not None:
    sys.path.insert(0, str(ROOT / "rl-strategy-optimizer" / "src" / "rl_brain_v2"))
import tape_feel as TF  # noqa: E402

TAPE = ROOT / "agent_work" / "kimi" / "tmp" / "xauusd_m1_vps_20220901_20261022.csv" if ROOT is not None else None
MANIFEST = HERE / "sample_manifest_20261022.csv"
SHOW, WARM, AHEAD = 72, 160, 26
BG, FG, GRID = "#121418", "#c8d0dc", "#2a2e36"


def h1_before(m1, anchor):
    seg = m1.iloc[max(0, m1.index.searchsorted(anchor - int((SHOW + WARM) * 3600 * 1.8))):m1.index.searchsorted(anchor)]
    g = seg.groupby(seg.index // 3600 * 3600)
    return pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last(),
                         "tick_volume": g.tick_volume.sum()}).tail(SHOW + WARM)


def panes(h):
    med = (h.high + h.low) / 2
    ema = h.close.ewm(span=20, adjust=False).mean()
    a10 = TF.atr(h, 10)
    sma, sd = h.close.rolling(20).mean(), h.close.rolling(20).std(ddof=0)
    tenkan = (h.high.rolling(9).max() + h.low.rolling(9).min()) / 2
    kijun = (h.high.rolling(26).max() + h.low.rolling(26).min()) / 2
    return {"bb_up": sma + 2 * sd, "bb_dn": sma - 2 * sd, "kc_up": ema + 2 * a10, "kc_dn": ema - 2 * a10,
            "jaw": TF.smma(med, 13), "teeth": TF.smma(med, 8), "lips": TF.smma(med, 5),
            "span_a": (tenkan + kijun) / 2, "span_b": (h.high.rolling(52).max() + h.low.rolling(52).min()) / 2,
            "obv": TF.obv(h), "ao": TF.ao(h), "ac": TF.ac(h)}


def bars_hist(ax, x, v):
    v = np.asarray(v, float)
    rising = np.r_[True, v[1:] >= v[:-1]]
    # PolyCollection instead of per-bar Rectangles (pixel-identical, much faster);
    # bar() datalim = [x0-0.4, xN+0.4] x [min(0,vmin), max(0,vmax)] — replicated below
    from matplotlib.collections import PolyCollection
    lo, hi = np.minimum(0, v), np.maximum(0, v)
    verts = [[(xi - 0.4, b), (xi + 0.4, b), (xi + 0.4, t), (xi - 0.4, t)]
             for xi, b, t in zip(x, lo, hi)]
    ax.add_collection(PolyCollection(
        verts, facecolors=list(np.where(rising, "#26b496", "#f05652")),
        edgecolors="none", zorder=1))
    ax.update_datalim([(x[0] - 0.4, lo.min()), (x[-1] + 0.4, hi.max())])
    ax.autoscale_view()
    ax.axhline(0, color=GRID, lw=0.8)


def render(h, path):
    p = panes(h)
    n = len(h)
    s = slice(n - SHOW, n)
    x = np.arange(SHOW)
    fig = plt.figure(figsize=(10, 8), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(4, 1, height_ratios=[5, 1.2, 1.2, 1.2], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    o, hi, lo, c = (h[k].to_numpy()[s] for k in ("open", "high", "low", "close"))
    col = np.where(c >= o, "#26b496", "#f05652")
    ax.vlines(x, lo, hi, color=col, lw=1)
    ax.bar(x, np.maximum(np.abs(c - o), 1e-6), bottom=np.minimum(o, c), width=0.7, color=col)
    ax.plot(x, p["bb_up"].to_numpy()[s], color="#8fa3c0", lw=0.9)
    ax.plot(x, p["bb_dn"].to_numpy()[s], color="#8fa3c0", lw=0.9)
    ax.plot(x, p["kc_up"].to_numpy()[s], color="#c08fbf", lw=0.9, ls="--")
    ax.plot(x, p["kc_dn"].to_numpy()[s], color="#c08fbf", lw=0.9, ls="--")
    for k, shift, cl in (("jaw", 8, "#3d7fe0"), ("teeth", 5, "#e04b4b"), ("lips", 3, "#3fc46a")):
        ax.plot(np.arange(SHOW + shift), p[k].to_numpy()[n - SHOW - shift:n], color=cl, lw=1.3)
    xa = np.arange(SHOW + AHEAD)
    cs = slice(n - SHOW - AHEAD, n)
    sa, sb = p["span_a"].to_numpy()[cs], p["span_b"].to_numpy()[cs]
    ax.fill_between(xa, sa, sb, where=sa >= sb, color="#2e8b57", alpha=0.18, lw=0)
    ax.fill_between(xa, sa, sb, where=sa < sb, color="#b22222", alpha=0.18, lw=0)
    lo_all = np.nanmin(np.r_[lo, p["bb_dn"].to_numpy()[s]])
    hi_all = np.nanmax(np.r_[hi, p["bb_up"].to_numpy()[s]])
    pad = (hi_all - lo_all) * 0.06
    ax.set_ylim(lo_all - pad, hi_all + pad)
    axes = [ax]
    for i, (k, kind) in enumerate((("obv", "line"), ("ao", "hist"), ("ac", "hist")), start=1):
        a = fig.add_subplot(gs[i], sharex=ax)
        v = p[k].to_numpy()[s]
        if kind == "line":
            a.plot(x, v, color="#e0b85a", lw=1.1)
        else:
            bars_hist(a, x, v)
        a.text(0.005, 0.8, k.upper(), transform=a.transAxes, color=FG, fontsize=8)
        a.set_yticks([])
        axes.append(a)
    for a in axes:
        a.set_facecolor(BG)
        a.grid(color=GRID, lw=0.5)
        a.tick_params(colors=FG, labelsize=8, labelbottom=False, bottom=False)
        for sp in a.spines.values():
            sp.set_color(GRID)
    ax.yaxis.tick_right()
    ax.set_xlim(-1, SHOW + AHEAD)
    ax.text(0.005, 0.97, "XAUUSD H1", transform=ax.transAxes, color=FG, fontsize=9, va="top")
    fig.savefig(path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "charts"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m1 = pd.read_csv(TAPE).set_index("time").sort_index()
    man = pd.read_csv(MANIFEST)
    if a.limit:
        man = man.head(a.limit)
    done = skipped = 0
    for r in man.itertuples():
        f = out / f"{r.sample_id}.png"
        if f.exists():
            continue
        h = h1_before(m1, int(r.anchor_tape))
        if len(h) < SHOW + 80:
            skipped += 1
            continue
        render(h, f)
        done += 1
    print(f"rendered {done}, skipped {skipped} (short history), dir {out}")


if __name__ == "__main__":
    main()
