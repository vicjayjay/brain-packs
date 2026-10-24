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
  EMBED        backbone only (512-d, projection head discarded), batch 1024.

Resume: every epoch checkpoints to Drive (ckpt_latest.pt); embedding writes
emb_panels_part_NNN.parquet + sha256 manifest, existing parts skipped.
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
EMB_BATCH = 1024
EMB_PART_ROWS = 50_000


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

    dl = DataLoader(_TrainDS(idx, train=True), batch_size=BATCH, shuffle=True,
                    num_workers=8, pin_memory=True, persistent_workers=True,
                    drop_last=True, collate_fn=_collate_train)
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
        torch.save(c, ckpt_f)
        print(f"epoch {ep} done, loss {tot/max(1,nb):.4f}, "
              f"{time.time()-t0:.0f}s, ckpt -> Drive", flush=True)
    torch.save({"bb": bb.state_dict()}, out / "encoder_final.pt")
    print("pretrain done -> encoder_final.pt", flush=True)


def embed(idx, out: Path):
    import torch
    from torch.utils.data import DataLoader
    dev = "cuda"
    bb, _ = _build_model()
    c = torch.load(out / "encoder_final.pt", map_location=dev, weights_only=False)
    bb.load_state_dict(c["bb"])
    bb.to(dev).eval()

    class _IdxDS(_PanelDS):
        def __getitem__(self, i):
            from PIL import Image
            e, p = self.idx[i]
            return e, self.tf(Image.open(p).convert("RGB"))

    def collate(batch):
        es = torch.tensor([b[0] for b in batch], dtype=torch.int64)
        xs = torch.stack([b[1] for b in batch])
        return es, xs

    dl = DataLoader(_IdxDS(idx, train=False), batch_size=EMB_BATCH, shuffle=False,
                    num_workers=8, pin_memory=True, persistent_workers=True,
                    collate_fn=collate)
    existing = {p.name for p in out.glob("emb_panels_part_*.parquet")}
    buf_e, buf_x = [], []
    part = 0
    man = {"files": {}}
    t0 = time.time()

    def flush():
        nonlocal part, buf_e, buf_x
        if not buf_e:
            return
        name = f"emb_panels_part_{part:03d}.parquet"
        part += 1
        if name in existing:
            print(f"skip {name} (exists)", flush=True)
            buf_e, buf_x = [], []
            return
        df = pd.DataFrame({"e": np.concatenate(buf_e)})
        X = np.concatenate(buf_x).astype(np.float32)
        for j in range(X.shape[1]):
            df[f"d{j}"] = X[:, j]
        df.to_parquet(out / name, index=False)
        h = hashlib.sha256((out / name).read_bytes()).hexdigest()
        man["files"][name] = {"rows": len(df), "sha256": h}
        print(f"{name}: {len(df):,} rows sha256 {h[:12]}...", flush=True)
        buf_e, buf_x = [], []

    with torch.no_grad():
        for es, xs in dl:
            xs = xs.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = bb(xs)
            buf_e.append(es.numpy())
            buf_x.append(z.float().cpu().numpy())
            if sum(len(b) for b in buf_e) >= EMB_PART_ROWS:
                flush()
    flush()
    man["n_rows"] = sum(f["rows"] for f in man["files"].values())
    man["elapsed_s"] = round(time.time() - t0, 1)
    (out / "emb_panels_manifest.json").write_text(json.dumps(man, indent=2))
    print(f"embed done: {man['n_rows']:,} rows in {len(man['files'])} parts", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pretrain", "embed", "all"])
    ap.add_argument("--drive-panels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-partial", action="store_true",
                    help="embed before the farm is complete (NOT judged-safe)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    drive_panels = Path(a.drive_panels)
    if a.cmd in ("embed", "all") and not a.allow_partial:
        # embedding parts are positional over the sorted index: running embed
        # before the farm is complete would freeze wrong part boundaries
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
        embed(idx, out)


if __name__ == "__main__":
    main()
