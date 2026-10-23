"""293k panel render farm — COLAB edition (chart-native-encoder arm, registered 2026-10-23).

Renders every row of the champion entry table (entries.parquet: e = bar-end epoch) as a
multi-TF panel JPEG into /content/panels/, zips each chunk to Drive, resumes from Drive
state. Multiprocessing over all vCPUs. Tape + entries come from the job1 release pack
(same tape, md5-verified 10-23).
"""
import argparse
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

WORK = Path("/content/panel_farm_work")
PANELS = Path("/content/panels")
STATE_NAME = "farm_done.txt"
CHUNK = 5000
TFS = {"M15": 15, "H1": 60, "H4": 240, "D1": 1440}
NEED = 232

_FRAMES = None
_SEARCH = None


def resample(m1: pd.DataFrame, minutes: int) -> pd.DataFrame:
    g = m1.groupby(m1.index // (minutes * 60) * (minutes * 60))
    return pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(),
                         "close": g.close.last(), "tick_volume": g.tick_volume.sum()})


def _init_frames(tape_pq: str):
    global _FRAMES, _SEARCH
    tape = pd.read_parquet(tape_pq)
    m1 = tape.set_index("time")[["open", "high", "low", "close", "tick_volume"]]
    m1.index = m1.index.astype(np.int64)
    _FRAMES = {tf: resample(m1, mins) for tf, mins in TFS.items()}
    _SEARCH = {tf: f.index.to_numpy() for tf, f in _FRAMES.items()}


def _render_one(e: int) -> tuple:
    from panel_situation_card import price_strip, SHOW
    from render_jay_template_20261022 import panes, BG, FG, GRID, bars_hist
    import matplotlib.pyplot as plt
    idx = {}
    for tf in _FRAMES:
        j = int(np.searchsorted(_SEARCH[tf], e, side="right"))
        if j < NEED:
            return e, "skip"
        idx[tf] = _FRAMES[tf].iloc[j - NEED:j]
    fig = plt.figure(figsize=(9, 12), dpi=80, facecolor=BG)
    gs = fig.add_gridspec(7, 1, height_ratios=[4, 4, 4, 4, 1, 1, 1], hspace=0.08)
    axes = []
    for i, tf in enumerate(("D1", "H4", "H1", "M15")):
        ax = fig.add_subplot(gs[i])
        price_strip(ax, idx[tf], f"XAUUSD {tf}")
        axes.append(ax)
    p1 = panes(idx["H1"])
    n = len(idx["H1"])
    s = slice(n - SHOW, n)
    x = np.arange(SHOW)
    for j, k in enumerate(("obv", "ao", "ac"), start=4):
        a = fig.add_subplot(gs[j], sharex=axes[0])
        v = p1[k].to_numpy()[s]
        if k == "obv":
            a.plot(x, v, color="#e0b85a", lw=1.0)
        else:
            bars_hist(a, x, v)
        a.set_yticks([])
        axes.append(a)
    for a in axes:
        a.set_facecolor(BG)
        a.grid(color=GRID, lw=0.5)
        a.tick_params(colors=FG, labelsize=7, labelbottom=False, bottom=False)
        for sp in a.spines.values():
            sp.set_color(GRID)
    out = PANELS / f"{int(e) % 40:02d}"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{int(e)}.jpg", facecolor=BG, bbox_inches="tight",
                pil_kwargs={"quality": 75})
    plt.close(fig)
    return e, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", default="/content/pack")
    ap.add_argument("--drive-dir", required=True)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    drive = Path(a.drive_dir)
    drive.mkdir(parents=True, exist_ok=True)
    PANELS.mkdir(exist_ok=True)

    _init_frames(str(Path(a.pack_dir) / "tape_xau_m1.parquet"))
    ent = pd.read_parquet(Path(a.pack_dir) / "entries.parquet")
    e_all = np.sort(ent["e"].to_numpy())
    state_f = drive / STATE_NAME
    done = set()
    if state_f.exists():
        done = {int(x) for x in state_f.read_text().split() if x.strip()}
    todo = [int(e) for e in e_all if int(e) not in done]
    print(f"total {len(e_all):,}, done {len(done):,}, todo {len(todo):,}, "
          f"procs {a.procs}", flush=True)

    t0 = time.time()
    n_ok = 0
    pending: list[int] = []
    part = len(list(drive.glob("panels_part_*.zip")))
    with state_f.open("a") as sf, Pool(a.procs) as pool:
        for i, (e, r) in enumerate(pool.imap_unordered(_render_one, todo, chunksize=8), 1):
            pending.append(e)
            n_ok += (r == "ok")
            if i % 250 == 0:
                dt = time.time() - t0
                eta = dt / i * (len(todo) - i) / 3600
                print(f"{i}/{len(todo)} ok={n_ok} {dt/i:.2f}s/panel ETA {eta:.1f}h",
                      flush=True)
            if len(pending) >= CHUNK:
                part += 1
                arc = shutil.make_archive(str(drive / f"panels_part_{part:03d}"),
                                          "zip", PANELS)
                sf.write("\n".join(str(x) for x in pending) + "\n")
                sf.flush()
                pending = []
                shutil.rmtree(PANELS)
                PANELS.mkdir()
                print(f"checkpoint zip -> {arc} (state flushed)", flush=True)
        if pending or any(PANELS.iterdir()):
            part += 1
            arc = shutil.make_archive(str(drive / f"panels_part_{part:03d}"), "zip", PANELS)
            sf.write("\n".join(str(x) for x in pending) + "\n")
            sf.flush()
    man = {"job": "panel_farm_job3", "total": len(e_all), "rendered_this_session": n_ok,
           "parts": part}
    (drive / "farm_manifest.json").write_text(json.dumps(man, indent=2))
    print(f"ALL DONE — {n_ok} panels this session, {part} part zips", flush=True)


if __name__ == "__main__":
    main()
