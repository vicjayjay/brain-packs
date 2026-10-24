"""panel_simclr_colab — job4: chart-native encoder, A100 self-supervised pretrain.

Arm: hypotheses.jsonl `chart-native-encoder` (registered 2026-10-23 BEFORE the render
farm started). This is step (2): SimCLR-style pretrain on the ~260k rendered multi-TF
panels, NO labels. The judge (step 3) stays home on the laptop.

PRE-COMMITTED choices (edit AFTER seeing results = p-hacking; changes need a new arm):
  MODEL        torchvision ResNet-18 (random init), fc -> Identity; SimCLR v1
               projection head MLP 512 -> 512 -> 128.
  INPUT        panels 720x960 -> resize 256 -> RandomResizedCrop 224.
  AUGMENT (chart-safe): RandomResizedCrop(scale 0.6-1.0), ColorJitter(b=0.25,c=0.25)
               p=0.8, GaussianBlur(5, 0.1-1.0) p=0.2. NO horizontal/vertical flip:
               the time axis is directional, a mirrored chart is not a valid chart.
  LOSS         NT-Xent, temperature 0.2, two views per image.
  OPT          AdamW lr 3e-4 wd 1e-4, cosine to 0, 1-epoch linear warmup, 20 epochs,
               batch 512, bf16 autocast.
  EMBED        backbone only (512-d, projection head discarded), bf16 autocast;
               default batch 512, <=4 workers, prefetch 2 to bound HOST RAM;
               optional batch override with CUDA-OOM halving (model unchanged).

One runtime owns pretraining. Any number of runtimes can then embed disjoint,
interleaved shards of the SAME frozen encoder. Deterministic 50k-row parts and
per-shard atomic manifests make resume independent of batch size. The last
runtime merges all complete shard manifests into the judge's original manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PANELS_LOCAL = Path("/content/panels")
EPOCHS = 20
BATCH = 512
LR = 3e-4
WD = 1e-4
TAU = 0.2
EMB_BATCH = 512
EMB_WORKERS = 4
EMB_PREFETCH = 2
EMB_PART_ROWS = 50_000


def shard_index(idx, shard: int, nshards: int):
    if nshards < 1 or not 0 <= shard < nshards:
        raise ValueError(f"need 0 <= shard < nshards, got {shard}/{nshards}")
    return idx[shard::nshards]


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    import torch
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _worker_count() -> int:
    return max(2, min(32, os.cpu_count() or 2))


def _loader_kwargs(workers: int, prefetch: int = 4) -> dict:
    kw = dict(num_workers=workers, pin_memory=True,
              persistent_workers=workers > 0)
    if workers > 0:
        kw["prefetch_factor"] = prefetch
    return kw


def _embed_resource_plan(workers: int | None, batch_size: int) -> tuple[int, int, int]:
    batch = batch_size or EMB_BATCH
    nworkers = min(EMB_WORKERS, _worker_count()) if workers is None else workers
    return batch, nworkers, EMB_PREFETCH


def merge_embedding_manifests(out: Path, nshards: int) -> dict:
    files = {}
    total = 0
    for shard in range(nshards):
        path = out / f"emb_panels_manifest_s{shard}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing {path.name}")
        man = json.loads(path.read_text())
        if not man.get("complete"):
            raise RuntimeError(f"shard {shard} is incomplete")
        if man.get("shard") != shard or man.get("nshards") != nshards:
            raise RuntimeError(f"shard identity mismatch in {path.name}")
        if man["n_rows"] != man["expected_rows"]:
            raise RuntimeError(f"row-count mismatch in {path.name}")
        for name, meta in man["files"].items():
            f = out / name
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            if h != meta["sha256"]:
                raise RuntimeError(f"sha256 mismatch on {name}")
            files[name] = meta
        total += man["n_rows"]
    merged = {"job": "panel_embed_job4", "nshards": nshards,
              "n_rows": total, "complete": True,
              "files": {name: files[name] for name in sorted(files)}}
    _atomic_json(out / "emb_panels_manifest.json", merged)
    return merged


def extract_panels(drive_panels: Path) -> list[tuple[int, Path]]:
    """Unzip every panels_part_NNN.zip to local SSD (Drive FUSE is too slow to
    serve 260k individual JPEGs). Marker file counts extracted zips -> resume."""
    PANELS_LOCAL.mkdir(exist_ok=True)
    zips = sorted(drive_panels.glob("panels_part_*.zip"))
    assert zips, f"no panel zips in {drive_panels}"
    marker = PANELS_LOCAL / ".extracted"
    n_done = int(marker.read_text()) if marker.exists() else 0
    for z in zips[n_done:]:
        print(f"extract {z.name}...", flush=True)
        for attempt in range(3):
            try:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(PANELS_LOCAL)
                break
            except OSError as ex:
                # Drive FUSE drops mid-read ("Transport endpoint is not
                # connected") when the mount blips; wait + retry, and if it
                # keeps failing tell the operator to re-run the mount cell
                if attempt == 2:
                    raise SystemExit(
                        f"Drive read failed on {z.name} ({ex}). "
                        "Re-run the mount cell above, then this cell — "
                        "extraction resumes at the failed zip.")
                print(f"  drive hiccup ({ex}), retry in 15s...", flush=True)
                time.sleep(15)
        n_done += 1
        marker.write_text(str(n_done))
    idx = []
    for p in PANELS_LOCAL.glob("*/*.jpg"):
        try:
            idx.append((int(p.stem), p))
        except ValueError:
            pass
    idx.sort()
    print(f"{len(idx):,} panels indexed from {n_done} zips", flush=True)
    return idx


def _build_model():
    import torch.nn as nn
    from torchvision.models import resnet18
    bb = resnet18(weights=None)
    bb.fc = nn.Identity()
    proj = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True),
                         nn.Linear(512, 128))
    return bb, proj


class _PanelDS:
    def __init__(self, idx, train: bool):
        from PIL import Image  # noqa: F401
        from torchvision import transforms as T
        self.idx = idx
        if train:
            self.tf = T.Compose([
                T.Resize(256),
                T.RandomResizedCrop(224, scale=(0.6, 1.0)),
                T.RandomApply([T.ColorJitter(brightness=0.25, contrast=0.25)], p=0.8),
                T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.2),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
            ])
        else:
            self.tf = T.Compose([
                T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
            ])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        from PIL import Image
        e, p = self.idx[i]
        return e, self.tf(Image.open(p).convert("RGB"))


class _TrainDS(_PanelDS):
    """Two independent augmented views per panel (SimCLR)."""

    def __getitem__(self, i):
        from PIL import Image
        e, p = self.idx[i]
        img = Image.open(p).convert("RGB")
        return self.tf(img), self.tf(img)


def _collate_train(batch):
    import torch
    v1 = torch.stack([b[0] for b in batch])
    v2 = torch.stack([b[1] for b in batch])
    return v1, v2


def _nt_xent(z1, z2, tau):
    import torch
    import torch.nn.functional as F
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(-9e15)
    targets = torch.arange(2 * n, device=z.device)
    targets = (targets + n) % (2 * n)
    return F.cross_entropy(sim, targets)


def pretrain(idx, out: Path):
    import torch
    from torch.utils.data import DataLoader
    dev = "cuda"
    bb, proj = _build_model()
    bb.to(dev)
    proj.to(dev)
    opt = torch.optim.AdamW(list(bb.parameters()) + list(proj.parameters()),
                            lr=LR, weight_decay=WD)
    steps_per_ep = math.ceil(len(idx) / BATCH)
    total = steps_per_ep * EPOCHS
    warm = steps_per_ep

    def lr_at(step):
        if step < warm:
            return LR * step / max(1, warm)
        t = (step - warm) / max(1, total - warm)
        return LR * 0.5 * (1 + math.cos(math.pi * t))

    ckpt_f = out / "ckpt_latest.pt"
    start_ep, step = 0, 0
    if ckpt_f.exists():
        c = torch.load(ckpt_f, map_location=dev, weights_only=False)
        bb.load_state_dict(c["bb"])
        proj.load_state_dict(c["proj"])
        opt.load_state_dict(c["opt"])
        start_ep, step = c["epoch"] + 1, c["step"]
        print(f"resume at epoch {start_ep} (step {step})", flush=True)

    workers = _worker_count()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    print(f"pretrain resource tuning: batch {BATCH} (registered), "
          f"workers {workers}, prefetch 4, bf16", flush=True)
    dl = DataLoader(_TrainDS(idx, train=True), batch_size=BATCH, shuffle=True,
                    drop_last=True, collate_fn=_collate_train,
                    **_loader_kwargs(workers))
    for ep in range(start_ep, EPOCHS):
        t0 = time.time()
        bb.train()
        proj.train()
        tot, nb = 0.0, 0
        for v1, v2 in dl:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            v1, v2 = v1.to(dev, non_blocking=True), v2.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z1 = proj(bb(v1))
                z2 = proj(bb(v2))
                loss = _nt_xent(z1.float(), z2.float(), TAU)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
            step += 1
            if nb % 100 == 0:
                print(f"ep{ep} {nb}/{steps_per_ep} loss {tot/nb:.4f} "
                      f"lr {lr_at(step):.2e}", flush=True)
        c = {"epoch": ep, "step": step, "bb": bb.state_dict(),
             "proj": proj.state_dict(), "opt": opt.state_dict()}
        _atomic_torch_save(c, ckpt_f)
        print(f"epoch {ep} done, loss {tot/max(1,nb):.4f}, "
              f"{time.time()-t0:.0f}s, ckpt -> Drive", flush=True)
    _atomic_torch_save({"bb": bb.state_dict()}, out / "encoder_final.pt")
    print("pretrain done -> encoder_final.pt", flush=True)


def embed(idx, out: Path, shard: int = 0, nshards: int = 1,
          workers: int | None = None, batch_size: int = 0):
    import torch
    from torch.utils.data import DataLoader
    dev = "cuda"
    (out / "emb_panels_manifest.json").unlink(missing_ok=True)
    encoder_f = out / "encoder_final.pt"
    while not encoder_f.exists():
        print("waiting for the single pretraining session to publish "
              "encoder_final.pt; retry in 60s...", flush=True)
        time.sleep(60)
    bb, _ = _build_model()
    c = torch.load(encoder_f, map_location=dev, weights_only=False)
    bb.load_state_dict(c["bb"])
    bb.to(dev).eval()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    class _IdxDS(_PanelDS):
        def __getitem__(self, i):
            from PIL import Image
            e, p = self.idx[i]
            return e, self.tf(Image.open(p).convert("RGB"))

    def collate(batch):
        es = torch.tensor([b[0] for b in batch], dtype=torch.int64)
        xs = torch.stack([b[1] for b in batch])
        return es, xs

    batch_size, workers, prefetch = _embed_resource_plan(workers, batch_size)
    print(f"embed shard {shard}/{nshards}: {len(idx):,} rows, batch {batch_size}, "
          f"workers {workers}, prefetch {prefetch}, bf16", flush=True)
    man_f = out / f"emb_panels_manifest_s{shard}.json"
    man = json.loads(man_f.read_text()) if man_f.exists() else {
        "job": "panel_embed_job4", "shard": shard, "nshards": nshards,
        "expected_rows": len(idx), "files": {}}
    if man.get("shard") != shard or man.get("nshards") != nshards:
        raise RuntimeError(f"shard identity mismatch in {man_f.name}")
    man.update(expected_rows=len(idx), complete=False)
    t0 = time.time()

    for part, start in enumerate(range(0, len(idx), EMB_PART_ROWS)):
        part_idx = idx[start:start + EMB_PART_ROWS]
        name = f"emb_panels_s{shard}_part_{part:03d}.parquet"
        f = out / name
        if f.exists():
            rows = len(pd.read_parquet(f, columns=["e"]))
            if rows != len(part_idx):
                raise RuntimeError(f"row-count mismatch in existing {name}")
            man["files"][name] = {"rows": rows,
                                  "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
            man["n_rows"] = sum(m["rows"] for m in man["files"].values())
            _atomic_json(man_f, man)
            print(f"skip {name} ({rows:,} rows, verified)", flush=True)
            continue

        while True:
            buf_e, buf_x = [], []
            try:
                dl = DataLoader(_IdxDS(part_idx, train=False), batch_size=batch_size,
                                shuffle=False, collate_fn=collate,
                                **_loader_kwargs(workers, prefetch))
                with torch.inference_mode():
                    for es, xs in dl:
                        xs = xs.to(dev, non_blocking=True)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            z = bb(xs)
                        buf_e.append(es.numpy())
                        buf_x.append(z.float().cpu().numpy())
                break
            except torch.cuda.OutOfMemoryError:
                del dl, buf_e, buf_x
                torch.cuda.empty_cache()
                if batch_size <= 256:
                    raise
                batch_size //= 2
                print(f"CUDA OOM; retry {name} with batch {batch_size}", flush=True)

        X = np.concatenate(buf_x).astype(np.float32)
        if X.shape != (len(part_idx), 512):
            raise RuntimeError(f"unexpected output shape for {name}: {X.shape}")
        df = pd.DataFrame(X, columns=[f"d{j}" for j in range(X.shape[1])])
        df.insert(0, "e", np.concatenate(buf_e))
        tmp = f.with_suffix(f".parquet.tmp-{os.getpid()}")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, f)
        man["files"][name] = {"rows": len(df),
                              "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        man["n_rows"] = sum(m["rows"] for m in man["files"].values())
        _atomic_json(man_f, man)
        print(f"{name}: {len(df):,} rows; shard total {man['n_rows']:,}/{len(idx):,}",
              flush=True)

    man["n_rows"] = sum(m["rows"] for m in man["files"].values())
    if man["n_rows"] != len(idx):
        raise RuntimeError(f"shard {shard} has {man['n_rows']} of {len(idx)} rows")
    man.update(complete=True, elapsed_s=round(time.time() - t0, 1))
    _atomic_json(man_f, man)
    print(f"embed shard {shard} done: {man['n_rows']:,} rows "
          f"in {len(man['files'])} parts", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pretrain", "embed", "all", "finalize"])
    ap.add_argument("--drive-panels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=-1,
                    help="data-loader workers; pretrain auto uses vCPUs, embed caps at 4")
    ap.add_argument("--embed-batch", type=int, default=0,
                    help="inference batch; 0 uses host-RAM-safe 512 with CUDA-OOM fallback")
    ap.add_argument("--allow-partial", action="store_true",
                    help="embed before the farm is complete (NOT judged-safe)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "finalize":
        man = merge_embedding_manifests(out, a.nshards)
        print(f"combined manifest ready: {man['n_rows']:,} rows, "
              f"{len(man['files'])} parts", flush=True)
        return
    if not a.drive_panels:
        ap.error("--drive-panels is required unless cmd=finalize")
    shard_index([], a.shard, a.nshards)
    drive_panels = Path(a.drive_panels)
    if a.cmd in ("embed", "all") and not a.allow_partial:
        if not (drive_panels / "FARM_COMPLETE.txt").exists():
            raise SystemExit(
                "refusing to embed: FARM_COMPLETE.txt not in the panels folder. "
                "Wait until every farm tab prints ALL DONE (the marker file is "
                "created then), or pass --allow-partial for a dry run that will "
                "NOT be used for the judge.")
    idx = extract_panels(drive_panels)
    if a.cmd in ("pretrain", "all"):
        pretrain(idx, out)
    if a.cmd in ("embed", "all"):
        embed(shard_index(idx, a.shard, a.nshards), out, a.shard, a.nshards,
              workers=None if a.workers < 0 else a.workers,
              batch_size=a.embed_batch)
        ready = True
        for shard in range(a.nshards):
            f = out / f"emb_panels_manifest_s{shard}.json"
            ready = ready and f.exists() and json.loads(f.read_text()).get("complete", False)
        if ready:
            man = merge_embedding_manifests(out, a.nshards)
            if man["n_rows"] != len(idx):
                raise RuntimeError(f"combined {man['n_rows']} rows != {len(idx)} panels")
            print(f"ALL SHARDS COMBINED: {man['n_rows']:,} rows; judge input ready",
                  flush=True)
        else:
            print("this shard is complete; waiting on other shard(s) before final merge",
                  flush=True)


if __name__ == "__main__":
    main()
