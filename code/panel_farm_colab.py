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


_FIG = None


def _worker_fig():
    """One figure per worker process, reused across renders (figure+axes
    construction was ~0.4s/panel). Callers must clear() every axis first;
    sharex links survive clear()."""
    global _FIG
    if _FIG is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from render_jay_template_20261022 import BG
        fig = plt.figure(figsize=(9, 12), dpi=80, facecolor=BG)
        gs = fig.add_gridspec(7, 1, height_ratios=[4, 4, 4, 4, 1, 1, 1],
                              hspace=0.08)
        axes = [fig.add_subplot(gs[i]) for i in range(4)]
        axes += [fig.add_subplot(gs[j], sharex=axes[0]) for j in (4, 5, 6)]
        _FIG = (fig, axes)
    return _FIG


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
    fig, axes = _worker_fig()
    for a in axes:
        a.clear()
    for i, tf in enumerate(("D1", "H4", "H1", "M15")):
        price_strip(axes[i], idx[tf], f"XAUUSD {tf}")
    p1 = panes(idx["H1"])
    n = len(idx["H1"])
    s = slice(n - SHOW, n)
    x = np.arange(SHOW)
    for j, k in enumerate(("obv", "ao", "ac"), start=4):
        a = axes[j]
        v = p1[k].to_numpy()[s]
        if k == "obv":
            a.plot(x, v, color="#e0b85a", lw=1.0)
        else:
            bars_hist(a, x, v)
        a.set_yticks([])
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
    return e, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", default="/content/pack")
    ap.add_argument("--drive-dir", required=True)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    assert 0 <= a.shard < a.nshards
    drive = Path(a.drive_dir)
    drive.mkdir(parents=True, exist_ok=True)
    if PANELS.exists():
        # leftovers from a crashed run: state file is the only truth, wipe
        shutil.rmtree(PANELS, ignore_errors=True)
    PANELS.mkdir(exist_ok=True)

    _init_frames(str(Path(a.pack_dir) / "tape_xau_m1.parquet"))
    ent = pd.read_parquet(Path(a.pack_dir) / "entries.parquet")
    e_all = np.sort(ent["e"].to_numpy())
    sharded = a.nshards > 1
    state_f = drive / (f"farm_done_s{a.shard}.txt" if sharded else STATE_NAME)
    done = set()
    for f in ([state_f, drive / STATE_NAME] if sharded else [state_f]):
        # sharded sessions also honor the legacy unsharded state (prior progress)
        if f.exists():
            done |= {int(x) for x in f.read_text().split() if x.strip()}
    # shard FIRST on the full sorted list (fixed universe per shard), then filter
    # done — filtering first would make the interleave depend on each session's
    # own progress and shards would overlap
    todo = [int(e) for e in e_all[a.shard::a.nshards] if int(e) not in done]
    print(f"total {len(e_all):,}, done {len(done):,}, todo {len(todo):,}, "
          f"procs {a.procs}, shard {a.shard}/{a.nshards}", flush=True)

    t0 = time.time()
    n_ok = 0
    pat = f"panels_part_s{a.shard}_*.zip" if sharded else "panels_part_*.zip"
    part = len(list(drive.glob(pat)))
    with state_f.open("a") as sf, Pool(a.procs) as pool:
        # batch-wise: the pool is fully drained BEFORE zip+rmtree, so no worker
        # can write into PANELS while it is archived/deleted (OSError 39 race,
        # and worse: files rendered during a zip used to land in the previous
        # chunk's archive, get deleted, then be marked done in state = lost)
        for start in range(0, len(todo), CHUNK):
            batch = todo[start:start + CHUNK]
            b_ok = 0
            for j, (e, r) in enumerate(
                    pool.imap_unordered(_render_one, batch, chunksize=8), 1):
                n_ok += (r == "ok")
                b_ok += (r == "ok")
                done_i = start + j
                if done_i % 250 == 0:
                    dt = time.time() - t0
                    eta = dt / done_i * (len(todo) - done_i) / 3600
                    print(f"{done_i}/{len(todo)} ok={n_ok} "
                          f"{dt/done_i:.2f}s/panel ETA {eta:.1f}h", flush=True)
            part += 1
            zname = (f"panels_part_s{a.shard}_{part:03d}" if sharded
                     else f"panels_part_{part:03d}")
            arc = shutil.make_archive(str(drive / zname), "zip", PANELS)
            sf.write("\n".join(str(x) for x in batch) + "\n")
            sf.flush()
            shutil.rmtree(PANELS)
            PANELS.mkdir()
            print(f"checkpoint zip -> {arc} ({b_ok}/{len(batch)} ok, "
                  f"state flushed)", flush=True)
    man = {"job": "panel_farm_job3", "total": len(e_all), "rendered_this_session": n_ok,
           "parts": part, "shard": a.shard, "nshards": a.nshards}
    mname = f"farm_manifest_s{a.shard}.json" if sharded else "farm_manifest.json"
    (drive / mname).write_text(json.dumps(man, indent=2))
    print(f"ALL DONE — {n_ok} panels this session, {part} part zips", flush=True)


if __name__ == "__main__":
    main()
