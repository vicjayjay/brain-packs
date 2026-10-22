# %% [markdown]
# # A100 lane job-1 — Kronos-mini embeddings of champion entry moments
#
# Registered: agent_work/brain/colab_a100_training_lane_20261022.md.
# Frozen representation only — NO training, NO evaluation here. The judge
# (5-fold time-blocked CV vs AUC 0.647) runs back home on the laptop.
#
# Rules: no secrets on Colab; checkpoints every chunk to Drive; resume-safe.
# Input pack: pack_job1/ (entries.parquet, tape_xau_m1.parquet,
# kronos_vendor_67b630e6.zip, pack_manifest.json).
# Set BRAIN_PACK_DIR to the Drive folder holding pack_job1/, or edit below.

# %% params
import os

# Pack comes from the public GitHub release (no Drive upload needed); outputs go to
# Drive (one in-notebook auth click) so they survive the session.
PACK_URL = os.environ.get(
    "BRAIN_PACK_URL",
    "https://github.com/vicjayjay/brain-packs/releases/download/job1-v1/pack_job1.zip")
PACK_DIR = os.environ.get("BRAIN_PACK_DIR", "")   # local override for smoke tests
OUT_DIR = "/content/drive/MyDrive/brain_packs/out_job1"
SPECS = {"h1_128": {"tf_minutes": 60, "len": 128},
         "m1_512": {"tf_minutes": 1, "len": 512}}
BATCH = 256
CHUNK = 10_000                                    # entries per checkpoint chunk
MODEL_REPO, TOK_REPO = "NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k"

# %% setup (Colab: run once; ~2 min incl. pack download)
import subprocess, sys, zipfile
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "huggingface_hub", "PyTorchModelHubMixin", "pandas", "pyarrow"], check=True)
try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")   # one consent click; session is already logged in
except ImportError:
    pass                                          # local run: set BRAIN_PACK_DIR

work = Path("/content/job1")
work.mkdir(exist_ok=True)
if PACK_DIR:                                      # local smoke test
    for f in ("entries.parquet", "tape_xau_m1.parquet", "pack_manifest.json",
              "kronos_vendor_67b630e6.zip"):
        (work / f).write_bytes((Path(PACK_DIR) / f).read_bytes())
else:
    if not (work / "pack_job1.zip").exists():
        subprocess.run(["curl", "-L", "-o", str(work / "pack_job1.zip"), PACK_URL], check=True)
    zipfile.ZipFile(work / "pack_job1.zip").extractall(work / "pack")
    src = work / "pack"
    if (src / "pack_job1").exists():
        src = src / "pack_job1"
    for f in ("entries.parquet", "tape_xau_m1.parquet", "pack_manifest.json",
              "kronos_vendor_67b630e6.zip"):
        (work / f).write_bytes((src / f).read_bytes())
if not (work / "kronos_vendor").exists():
    zipfile.ZipFile(work / "kronos_vendor_67b630e6.zip").extractall(work / "kronos_vendor")
sys.path.insert(0, str(work / "kronos_vendor"))
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
from model.kronos import Kronos, KronosTokenizer   # vendored, pinned 67b630e6

tok = KronosTokenizer.from_pretrained(TOK_REPO)
model = Kronos.from_pretrained(MODEL_REPO)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
tok.to(dev).eval()
model.to(dev).eval()
print("device:", dev, "| d_model:", model.d_model, "| layers:", model.n_layers)

# %% load pack + build window index
entries = pd.read_parquet(work / "entries.parquet")          # e = bar-END epoch (tape frame)
tape = pd.read_parquet(work / "tape_xau_m1.parquet")
tape["dt"] = pd.to_datetime(tape["time"], unit="s")

frames = {"m1_512": tape.reset_index(drop=True)}
h1 = tape.set_index("dt").resample("1h", label="right", closed="right").agg(
    {"open": "first", "high": "max", "low": "min", "close": "last",
     "tick_volume": "sum"}).dropna().reset_index()
h1["time"] = (h1["dt"].astype("datetime64[ns]").view("int64") // 10**9).astype("int64")
frames["h1_128"] = h1.reset_index(drop=True)
print({k: len(v) for k, v in frames.items()})

PRICE_COLS = ["open", "high", "low", "close"]


def window_batch(fr: pd.DataFrame, ends: np.ndarray, length: int, bar_min: int):
    """Last `length` bars with bar-end <= e for each e (searchsorted, no lookahead).
    Returns x [B,L,6] z-scored+clipped per window (upstream convention), stamps [B,L,5],
    and a mask of entries with enough history."""
    tend = fr["time"].to_numpy() + 60 * bar_min                                 # bar END epoch
    pos = np.searchsorted(tend, ends, side="right")                            # bars with end <= e
    ok = pos >= length
    idx = pos[ok, None] - length + np.arange(length)[None, :]
    w = fr.iloc[idx.ravel()]
    ohlc = w[PRICE_COLS].to_numpy(np.float32).reshape(len(idx), length, 4)
    vol = w["tick_volume"].to_numpy(np.float32).reshape(len(idx), length, 1)
    amt = vol * ohlc.mean(axis=2, keepdims=True)
    x = np.concatenate([ohlc, vol, amt], axis=2)
    mu, sd = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True)
    x = np.clip((x - mu) / (sd + 1e-5), -5, 5)
    dt_flat = pd.to_datetime(tend[idx.ravel()], unit="s")   # pandas needs 1-D
    st = np.stack([dt_flat.minute.to_numpy(), dt_flat.hour.to_numpy(),
                   dt_flat.dayofweek.to_numpy(), dt_flat.day.to_numpy(),
                   dt_flat.month.to_numpy()], axis=1
                  ).reshape(len(idx), length, 5).astype(np.float32)
    return x, st, ok


# %% embed loop (chunked checkpoints; resume skips finished chunks)
import time

for spec, cfg in SPECS.items():
    fr = frames[spec]
    e_all = entries["e"].to_numpy()
    n_chunks = int(np.ceil(len(e_all) / CHUNK))
    t0 = time.time()
    for ci in range(n_chunks):
        out_f = Path(OUT_DIR) / f"chunk_{ci:04d}_{spec}.parquet"
        if out_f.exists():
            continue
        lo, hi = ci * CHUNK, min((ci + 1) * CHUNK, len(e_all))
        es, ys = e_all[lo:hi], entries["y"].to_numpy()[lo:hi]
        embs, keep_e, keep_y = [], [], []
        for b in range(0, len(es), BATCH):
            x, st, ok = window_batch(fr, es[b:b + BATCH], cfg["len"], cfg["tf_minutes"])
            if not ok.any():
                continue
            xt = torch.from_numpy(x[ok]).to(dev)
            stt = torch.from_numpy(st[ok]).to(dev)
            with torch.no_grad():
                s1, s2 = tok.encode(xt, half=True)
                _, ctx = model.decode_s1(s1, s2, stamp=stt)     # [B,L,d_model] post-norm
                emb = ctx.mean(dim=1).half().cpu().numpy()      # mean-pool over seq
            embs.append(emb)
            keep_e.append(es[b:b + BATCH][ok])
            keep_y.append(ys[b:b + BATCH][ok])
        if embs:
            E = np.concatenate(embs)
            df = pd.DataFrame(E, columns=[f"d{i}" for i in range(E.shape[1])])
            df.insert(0, "e", np.concatenate(keep_e))
            df.insert(1, "y", np.concatenate(keep_y))
            df.to_parquet(out_f, index=False)
        print(f"{spec} chunk {ci + 1}/{n_chunks} done ({time.time() - t0:.0f}s)", flush=True)

# %% finalize: merge chunks + manifest
import hashlib, json

for spec in SPECS:
    chunks = sorted(Path(OUT_DIR).glob(f"chunk_*_{spec}.parquet"))
    full = pd.concat([pd.read_parquet(c) for c in chunks], ignore_index=True)
    f = Path(OUT_DIR) / f"emb_{spec}.parquet"
    full.to_parquet(f, index=False)
    print(spec, full.shape, "sha256", hashlib.sha256(f.read_bytes()).hexdigest()[:16])

run_man = dict(model=MODEL_REPO, tokenizer=TOK_REPO, vendor="shiyu-coder/Kronos@67b630e6",
               specs=SPECS, batch=BATCH, pooling="mean over seq of decode_s1 context (post-norm)",
               dtype="float16", created=pd.Timestamp.utcnow().isoformat())
Path(OUT_DIR, "run_manifest.json").write_text(json.dumps(run_man, indent=2))
print("DONE — artifacts in", OUT_DIR)
