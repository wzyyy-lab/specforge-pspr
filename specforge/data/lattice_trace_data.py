"""Anchor-level lattice trace data plane, ported verbatim from the dh2048 reference recipe.

The reference training run (``TAPS-SP/scripts/train_accept_selector.py``, checkpoint
``TAPS-SP/outputs/dh2048/best.pt``) does NOT run the draft backbone during selector training. Every
anchor record already carries the pre-extracted lattice: ``top_token_ids[H,K]``,
``top_log_probs[H,K]``, ``draft_hidden[H,Dh]``, the committed continuation ``g_block[H]`` and three
uncertainty scalars. This module reproduces that reader bit for bit so the ported trainer sees the
same tensors as the reference.

Provenance of each function:
  ``align_top1``, ``precompute``, ``load_vocab_embeds``  <- TAPS-SP/scripts/train_pairwise.py
  ``collate``                                            <- TAPS-SP/scripts/train_lattice_selector.py
  ``load_files``, ``is_cal``, ``dataset_files``          <- TAPS-SP/scripts/train_accept_selector.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zlib import crc32

import torch

__all__ = [
    "align_top1",
    "precompute",
    "collate",
    "load_vocab_embeds",
    "load_files",
    "is_cal",
    "dataset_files",
]


def load_vocab_embeds(model_path: str) -> torch.Tensor:
    model_dir = Path(model_path)
    index_file = model_dir / "model.safetensors.index.json"
    from safetensors.torch import load_file

    if index_file.exists():
        index = json.loads(index_file.read_text())
        shard = index["weight_map"]["model.embed_tokens.weight"]
        weights = load_file(str(model_dir / shard))
    else:
        weights = load_file(str(model_dir / "model.safetensors"))
    return weights["model.embed_tokens.weight"]


def align_top1(rec):
    """Make candidate column 0 be the token DFlash ACTUALLY drafts.

    lm_head runs in bf16, so the maximal logit is EXACTLY tied across tokens at 2.92% of slots, and
    torch.topk breaks such ties differently from torch.argmax (measured: 33% of tied cases).  Column
    0 therefore disagreed with `drafted_top1` at 2.92% of slots, which (a) put that much label noise
    on the single most important signal ("is top-1 already right?") and (b) broke the tau=inf ==
    DFlash guarantee at decode.  The tied entries have IDENTICAL log-probs (verified 4289/4289), so
    swapping columns leaves top_log_probs / entropy / margin / mass exactly unchanged.
    """
    d1 = rec["drafted_top1"].long()  # [H]
    top = rec["top_token_ids"].long()  # [H,K]
    eq = top == d1.unsqueeze(1)
    j = torch.where(eq.any(dim=1), eq.long().argmax(dim=1), torch.zeros_like(d1))
    if bool((j != 0).any()):
        ar = torch.arange(top.shape[0])
        for name in ("top_token_ids", "top_log_probs"):
            v = rec[name].clone()
            c0 = v[:, 0].clone()
            v[:, 0] = v[ar, j]
            v[ar, j] = c0
            rec[name] = v
    return rec


def precompute(rec):
    """Add target_idx[H], oracle_len, L0, OK16 to a record (tensors on cpu)."""
    align_top1(rec)
    top = rec["top_token_ids"].long()  # [H,K]
    g = rec["g_block"].long()  # [H]
    H, K = top.shape
    valid = g >= 0
    eqK = (top == g.unsqueeze(1)) & valid.unsqueeze(1)  # [H,K]
    has = eqK.any(dim=1)
    idx = torch.where(has, eqK.float().argmax(dim=1), torch.full((H,), -1, dtype=torch.long))
    rec["target_idx"] = idx
    # oracle_len = leading run where g in top-K (idx>=0)
    m = 0
    for t in range(H):
        if idx[t] >= 0:
            m += 1
        else:
            break
    rec["oracle_len"] = m
    # L0 = leading run where drafted_top1 == g  (== top1 candidate == top[:,0])
    L0 = 0
    for t in range(H):
        if valid[t] and top[t, 0] == g[t]:
            L0 += 1
        else:
            break
    rec["L0"] = L0
    rec["OK16"] = m  # top-16 oracle (K=16 traces)
    return rec


def collate(batch, embed, device):
    top = torch.stack([b["top_token_ids"].long() for b in batch]).to(device)  # [B,H,K]
    lp = torch.stack([b["top_log_probs"].float() for b in batch]).to(device)  # [B,H,K]
    dh = torch.stack([b["draft_hidden"].float() for b in batch]).to(device)  # [B,H,Dh]
    g = torch.stack([b["g_block"].long() for b in batch]).to(device)  # [B,H]
    ti = torch.stack([b["target_idx"].long() for b in batch]).to(device)  # [B,H]
    scal = torch.stack(
        [
            torch.stack(
                [
                    b["position_entropy"].float(),
                    b["top1_top2_margin"].float().clamp(-20, 20),
                    b["topk_mass"].float(),
                ],
                dim=-1,
            )
            for b in batch
        ]
    ).to(device)  # [B,H,3]
    anchor = torch.tensor([b["anchor_tok"] for b in batch], dtype=torch.long, device=device)
    B, H = g.shape
    prefix = torch.cat([anchor.view(B, 1), g[:, :-1].clamp(min=0)], dim=1)  # [B,H]
    prefix_emb = embed[prefix]
    cand_emb = embed[top]
    out = dict(
        top=top,
        lp=lp,
        dh=dh,
        g=g,
        ti=ti,
        scal=scal,
        prefix_ids=prefix,
        prefix_emb=prefix_emb,
        cand_emb=cand_emb,
    )
    if all("anchor_thidden" in b for b in batch):  # trace dirs may be mixed
        out["th"] = torch.stack([b["anchor_thidden"].float() for b in batch]).to(device)  # [B,5*Dh]
    return out


def load_files(files, keep=None, workers=16):
    """Load trace files in parallel. Each file holds up to 512 anchor records spanning MANY prompts,
    so `keep(prompt_id)` filters at load time (keeps memory small and the split exact)."""

    def one(f):
        rs = torch.load(f, weights_only=False)
        out = []
        for r in rs:
            if keep is not None and not keep(r["prompt_id"]):
                continue
            precompute(r)
            if "draft_hidden" in r:
                out.append(r)
        return out

    out = []
    with ThreadPoolExecutor(workers) as ex:
        for rs in ex.map(one, files):
            out.extend(rs)
    return out


def _ds_of(f: Path):
    """dataset name from the shard dir (outputs/seqtrace_train/<ds>[_<offset>]/xxx.pt)."""
    return re.sub(r"_\d+$", "", f.parent.name)


def is_cal(prompt_id, val_pct, seed):
    """Stable, process-independent PROMPT-level split.  Files span multiple prompts (and 11% of
    prompts straddle file boundaries), so splitting by file WOULD LEAK -- hash the prompt id."""
    return crc32(f"{seed}:{prompt_id}".encode()) % 1000 < val_pct * 10


def dataset_files(trace_dirs):
    """trace_dirs may be a comma-separated list, so off-policy and on-policy (DAgger) traces can be
    AGGREGATED rather than replaced -- training only on the newest policy's anchors is known to be
    unstable, aggregation is what makes DAgger work."""
    by_ds = defaultdict(list)
    for d in str(trace_dirs).split(","):
        d = d.strip()
        if not d:
            continue
        for f in sorted(Path(d).rglob("*.pt")):
            by_ds[_ds_of(f)].append(f)
    return by_ds
