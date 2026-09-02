#!/usr/bin/env python3
"""Gate for target-greedy selector labels: the sidecar path and, above all, the alignment.

The alignment is the whole risk. ``target_greedy`` is stored shifted by one against ``input_ids``,
so an off-by-one in the gather is not a crash and not even a visible regression -- it is a plausible
label set that trains a selector to predict the *previous* token. That reads, from the outside, as
"the idea did not work". So this gate does not check the gather against the dump's own convention;
it recomputes the target model's argmax from scratch and requires the label the objective actually
uses to equal ``argmax P(. | input_ids[:anchor + k])`` for every supervised slot.

Properties:

1. **Sidecar plumbing.** Reader discovers the sidecar, store merges it, normalizer passes it
   through, collator emits it. A missing sidecar file fails loudly at assembly, and a batch where
   only some samples carry the key is rejected rather than silently collapsing to the corpus labels.

2. **Alignment against the live target model.** For every supervised slot of a real sample, the
   label handed to the selector equals the target's own greedy continuation of that slot's exact
   prefix. Checked at the value level, not by re-deriving the same index arithmetic.

3. **Flag off is bit-identical.** With the flag off, a batch that carries the sidecar key and one
   that does not must produce the same loss and the same metrics to the last bit -- otherwise the
   feature is not opt-in and every DFlash2 result before it is in question.

4. **Only the labels move.** With the flag on, the backbone CE numerator, the token accuracy and the
   loss denominator must be unchanged; only the selector terms may differ. Compared at the chunk
   level on the ``ce_loss_num`` field, because the published ``loss_terms`` is the *composed* total
   -- backbone CE plus the selector and detector contributions -- so it is expected to move and
   would make this check vacuous in the wrong direction.

5. **The labels really do differ**, and the gate can fail: a deliberately mis-shifted label tensor
   must break check 2.

Usage:
    PYTHONPATH=. python scripts/gate_target_greedy_labels.py \
        --features cache/hidden_states/sharegpt3k \
        --target-model /kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

SPECFORGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPECFORGE))

from specforge.algorithms.common.collation import (  # noqa: E402
    pad_and_concatenate_features,
)
from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel  # noqa: E402
from specforge.algorithms.common.hidden_states_data import (  # noqa: E402
    TARGET_GREEDY_KEY,
    build_collator,
    build_offline_reader,
    normalize_offline_sample,
    target_greedy_sidecar_dir,
)
from specforge.modeling.draft.pspr import PSPRDraftModel  # noqa: E402
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead  # noqa: E402
from specforge.runtime.data_plane.feature_store import LocalFeatureStore  # noqa: E402
from specforge.runtime.data_plane.offline_reader import list_feature_files  # noqa: E402
from specforge.training.model_loading import warm_start_draft_model  # noqa: E402


def build_draft(draft_config, device, dtype, backbone):
    from transformers import Qwen3Config

    raw = json.load(open(draft_config))
    cfg = Qwen3Config(**raw)
    cfg.dflash_config = raw["dflash_config"]
    cfg.block_size = raw["block_size"]
    cfg.num_target_layers = raw["num_target_layers"]
    cfg._attn_implementation = "flex_attention"
    torch.manual_seed(0)
    with torch.device("cpu"):
        draft = PSPRDraftModel(cfg)
    # Warm start is load-bearing: on a random backbone the target almost never lands in the top-16,
    # so selector coverage is 0 and every selector-side comparison degenerates to 0 == 0.
    report = warm_start_draft_model(
        draft,
        backbone,
        draft_config=cfg,
        strategy="offline",
        allow_missing_embedding=True,
    )
    print(f"warm start: {report.loaded_keys} tensors from {Path(backbone).name}")
    return draft.to(device=device, dtype=dtype), cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(SPECFORGE / "cache/hidden_states/sharegpt3k"))
    ap.add_argument("--draft-config", default=str(SPECFORGE / "configs/qwen3-4b-pspr.json"))
    ap.add_argument("--target-model", required=True)
    ap.add_argument(
        "--backbone",
        default="/kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B-DFlash-b16",
    )
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()

    device, dtype = torch.device("cuda"), torch.bfloat16
    checks: dict[str, bool] = {}

    # ---- 1: sidecar plumbing ------------------------------------------------------------------
    print("=" * 78)
    print("1. sidecar plumbing")
    sidecar = target_greedy_sidecar_dir(args.features)
    print(f"   sidecar dir             : {sidecar}")
    if sidecar is None:
        print(f"   FAIL: expected {args.features}.target_greedy to exist")
        return 1
    reader = build_offline_reader(
        "dflash", args.features, run_id="gate", ttt_length=7, max_len=args.max_len
    )
    checks["reader requests the sidecar key"] = TARGET_GREEDY_KEY in reader.feature_keys
    store = LocalFeatureStore("gate")
    raw_batch = []
    # Truncation at ``max_len`` can leave a sample without two consecutive supervised tokens, which
    # the DFlash normalizer rejects. Skip those rather than pick a length that happens to work.
    for ref in reader.read(limit=64):
        tensors, handle = store.get(ref)
        try:
            raw_batch.append(normalize_offline_sample(tensors, args.max_len))
        except ValueError:
            pass
        finally:
            store.release(handle)
        if len(raw_batch) == 2:
            break
    if len(raw_batch) < 2:
        print("   FAIL: could not find two usable samples")
        return 1
    checks["store merges the sidecar"] = all(
        TARGET_GREEDY_KEY in feature for feature in raw_batch
    )
    collate = build_collator()
    batch = collate(raw_batch)
    print(f"   collated keys           : {sorted(batch)}")
    print(
        f"   target_greedy           : {tuple(batch[TARGET_GREEDY_KEY].shape)} "
        f"{batch[TARGET_GREEDY_KEY].dtype}"
    )
    checks["collator emits the sidecar key"] = (
        TARGET_GREEDY_KEY in batch
        and batch[TARGET_GREEDY_KEY].shape == batch["input_ids"].shape
    )

    # A dump with no sidecar must behave exactly as before: no key anywhere.
    with tempfile.TemporaryDirectory() as tmp:
        bare = Path(tmp) / "bare"
        bare.mkdir()
        files = list_feature_files(args.features)
        (bare / Path(files[0]).name).symlink_to(files[0])
        bare_reader = build_offline_reader(
            "dflash", str(bare), run_id="bare", ttt_length=7, max_len=args.max_len
        )
        checks["no sidecar dir -> key never requested"] = (
            TARGET_GREEDY_KEY not in bare_reader.feature_keys
            and bare_reader.sidecar_dir is None
            and "sidecar_path" not in bare_reader.read(limit=1)[0].metadata
        )
        # A half-built sidecar directory is the dangerous case: it would train part of the run on
        # one label set and part on another. It must fail at assembly.
        partial = Path(tmp) / "partial"
        partial.mkdir()
        for path in files[:2]:
            (partial / Path(path).name).symlink_to(path)
        (Path(str(partial) + ".target_greedy")).mkdir()
        stem = Path(files[0]).name[: -len(".ckpt")]
        (Path(str(partial) + ".target_greedy") / (stem + ".pt")).symlink_to(
            Path(sidecar).resolve() / (stem + ".pt")
        )
        try:
            build_offline_reader(
                "dflash", str(partial), run_id="p", ttt_length=7, max_len=args.max_len
            ).read()
            checks["partial sidecar dir is rejected"] = False
        except FileNotFoundError as exc:
            print(f"   partial sidecar         : rejected -> {type(exc).__name__}")
            checks["partial sidecar dir is rejected"] = True

    # Mixed batches are a dataset bug, not a supported mode.
    try:
        collate([raw_batch[0], {k: v for k, v in raw_batch[1].items() if k != TARGET_GREEDY_KEY}])
        checks["mixed batch is rejected"] = False
    except KeyError:
        checks["mixed batch is rejected"] = True

    # ---- model / target ------------------------------------------------------------------------
    draft, cfg = build_draft(args.draft_config, device, dtype, args.backbone)
    parts = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model, device="cuda", dtype=dtype
    )
    draft.bind_target_decoder(parts.embed_tokens)

    def build_model(greedy_labels: bool):
        return (
            OnlineDFlashModel(
                draft_model=draft,
                target_lm_head=parts.lm_head,
                target_embed_tokens=parts.embed_tokens,
                mask_token_id=cfg.dflash_config["mask_token_id"],
                block_size=cfg.block_size,
                attention_backend="flex_attention",
                num_anchors=64,
                loss_decay_gamma=None,
                selector_loss_alpha=1.0,
                selector_err_loss_alpha=1.0,
                selector_own_denominator=True,
                selector_target_greedy_labels=greedy_labels,
            )
            .to(device=device, dtype=dtype)
            .eval()
        )

    single = collate(raw_batch[:1])
    inputs = dict(
        input_ids=single["input_ids"].to(device),
        hidden_states=single["hidden_states"].to(device=device, dtype=dtype),
        loss_mask=single["loss_mask"].to(device),
    )
    greedy = single[TARGET_GREEDY_KEY].to(device)

    # ---- 2: alignment against the live target model --------------------------------------------
    print("\n2. label alignment, recomputed from the target model")
    from transformers import AutoModelForCausalLM

    target = (
        AutoModelForCausalLM.from_pretrained(
            args.target_model, attn_implementation="sdpa", dtype=dtype
        )
        .to(device)
        .eval()
    )
    with torch.no_grad():
        true_logits = target(inputs["input_ids"]).logits[0]
    # ``fresh[p]`` = the target's greedy token *for* position p, recomputed here with no reference
    # to how the dump chose to store it.
    fresh = torch.empty_like(inputs["input_ids"][0])
    fresh[0] = 0
    fresh[1:] = true_logits[:-1].argmax(dim=-1)
    del true_logits, target
    torch.cuda.empty_cache()

    model = build_model(True)
    torch.manual_seed(7)
    with torch.no_grad():
        anchors, keep, _ = model._forward_draft_blocks(
            input_ids=inputs["input_ids"],
            hidden_states=inputs["hidden_states"],
            loss_mask=inputs["loss_mask"],
            max_valid_anchors=None,
        )
    block = cfg.block_size
    seq_len = inputs["input_ids"].shape[1]
    offsets = torch.arange(block, device=device).view(1, 1, -1)
    label_indices = anchors.unsqueeze(-1) + offsets
    safe = label_indices.clamp(max=seq_len - 1)
    labels = model._selector_target_ids(target_greedy=greedy, safe_label_indices=safe)

    supervised = (
        keep.unsqueeze(-1).expand(-1, -1, block)
        & (label_indices < seq_len)
        & (offsets > 0)
        & (
            torch.gather(
                inputs["loss_mask"].unsqueeze(1).expand(-1, anchors.size(1), -1), 2, safe
            )
            > 0
        )
    )
    expected = fresh[safe.clamp(max=seq_len - 1)]
    n_sup = int(supervised.sum())
    agree = int((labels[supervised] == expected[supervised]).sum())
    corpus = torch.gather(
        inputs["input_ids"].unsqueeze(1).expand(-1, anchors.size(1), -1), 2, safe
    )
    differ = int((labels[supervised] != corpus[supervised]).sum())
    print(f"   supervised slots        : {n_sup}")
    print(f"   label == target argmax  : {agree}/{n_sup}")
    print(f"   label != corpus token   : {differ}/{n_sup}  ({differ / max(n_sup, 1):.4f})")
    checks["labels equal the target's own argmax"] = n_sup > 0 and agree == n_sup
    checks["labels actually differ from corpus"] = differ > 0

    # 5a: the gate must be able to fail. A one-slot mis-shift is the realistic bug.
    mis = model._selector_target_ids(
        target_greedy=greedy, safe_label_indices=(safe - 1).clamp_min(0)
    )
    mis_agree = int((mis[supervised] == expected[supervised]).sum())
    print(f"   mis-shifted control     : {mis_agree}/{n_sup} agree (must be < all)")
    checks["gate detects a mis-shift"] = mis_agree < n_sup

    # ---- 3 / 4: flag off is bit-identical; flag on moves only the selector --------------------
    print("\n3. flag off is bit-identical / 4. only the selector terms move")

    def run(model_, with_key):
        torch.manual_seed(7)
        with torch.no_grad():
            loss, acc, metrics = model_(
                **inputs, target_greedy=greedy if with_key else None
            )
        flat = {"loss": float(loss), "accuracy": float(acc)}
        for name, (num, den) in metrics["ratio_metrics"].items():
            flat[name] = (float(num), float(den))
        flat["loss_terms"] = tuple(float(x) for x in metrics["loss_terms"])
        return flat

    off = build_model(False)
    off_without = run(off, False)
    off_with = run(off, True)
    identical = [k for k in off_without if off_without[k] != off_with[k]]
    print(f"   flag off, key present vs absent: differing entries = {identical}")
    checks["flag off ignores the sidecar bit-for-bit"] = not identical

    on_with = run(model, True)
    # ``loss_terms`` is deliberately NOT in this list: it carries the composed numerator, which
    # already includes the selector CE and detector contributions, so it must move when the selector
    # labels move. The backbone-only quantities are isolated below at the chunk level instead.
    frozen = ["accuracy", "target_probability"]
    drift = {k: (off_with[k], on_with[k]) for k in frozen if off_with[k] != on_with[k]}
    print(f"   backbone metrics unchanged: {not drift}")
    for k, (a, b) in drift.items():
        print(f"     {k}: {a} -> {b}")
    same_den = off_with["loss_terms"][1] == on_with["loss_terms"][1]
    print(f"   loss denominator unchanged: {same_den}")

    # The decisive form: the backbone CE numerator is a separate field of the chunk terms, so it can
    # be compared directly instead of reconstructed out of the composed loss.
    torch.manual_seed(7)
    with torch.no_grad():
        anchors2, keep2, out_hidden = model._forward_draft_blocks(
            input_ids=inputs["input_ids"],
            hidden_states=inputs["hidden_states"],
            loss_mask=inputs["loss_mask"],
            max_valid_anchors=None,
        )
        hidden_4d = out_hidden.reshape(1, anchors2.shape[1], block, -1)
        target_ids = corpus
        pred_ids = torch.cat([target_ids[:, :, :1], target_ids[:, :, :-1]], dim=-1)
        wm = (
            keep2.unsqueeze(-1).expand(-1, -1, block).float()
            * (label_indices < seq_len).float()
            * (offsets > 0).float()
            * torch.gather(
                inputs["loss_mask"].unsqueeze(1).expand(-1, anchors2.size(1), -1), 2, safe
            )
        )
        corpus_terms = model._dflash_objective_chunk_terms(
            hidden_4d, target_ids, wm, pred_ids, None
        )
        greedy_terms = model._dflash_objective_chunk_terms(
            hidden_4d, target_ids, wm, pred_ids, labels
        )
    ce_delta = abs(float(corpus_terms.ce_loss_num) - float(greedy_terms.ce_loss_num))
    sel_delta = abs(float(corpus_terms.selector_ce_num) - float(greedy_terms.selector_ce_num))
    print(f"   chunk ce_loss_num     : |delta| = {ce_delta:.3e}   (backbone, must be 0)")
    print(f"   chunk selector_ce_num : |delta| = {sel_delta:.3e}   (selector, must be > 0)")
    checks["flag on leaves the backbone objective untouched"] = (
        not drift and same_den and ce_delta == 0.0
    )
    checks["the selector objective really changes"] = sel_delta > 0.0
    moved = [
        k
        for k in ("selector_loss", "selector_accuracy", "selector_coverage")
        if off_with[k] != on_with[k]
    ]
    print(f"   selector-side entries that moved: {moved}")
    checks["flag on changes the selector objective"] = len(moved) == 3

    def ratio(flat, name):
        num, den = flat[name]
        return num / max(den, 1.0)

    print("\n   selector metrics, corpus labels -> greedy labels")
    for name in ("selector_loss", "selector_accuracy", "selector_coverage"):
        print(f"     {name:20s} {ratio(off_with, name):.4f} -> {ratio(on_with, name):.4f}")

    # ---- verdict ------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    n_ok = sum(checks.values())
    print(f"\n{n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
