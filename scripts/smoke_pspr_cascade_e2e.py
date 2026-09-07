#!/usr/bin/env python3
"""Tiny end-to-end smoke for PSPR-cascade: warm start -> update -> export -> reload -> decode.

Every stage in this chain has already failed silently at least once in this project's history, and
none of them is covered by the unit-level gate:

  * the warm start fails CLOSED on a champion checkpoint unless the cascade tensors are declared as an
    all-or-nothing optional key group (this was a FATAL);
  * the exporter can label a cloze checkpoint ``CascadeCorrector`` and only trip much later;
  * the external decoder builds the selector from the artefact's ``config`` dict, so a key the exporter
    forgets becomes a constructor DEFAULT -- and two of those defaults (``err_score_enabled``,
    ``err_score_anchor_rho``) change the decision rule while every tensor still loads strictly;
  * native serving and external serving are two separate transcriptions of the same walk.

So this runs the real code for all of it at toy scale: the real ``warm_start_draft_model``, the real
``apply_backbone_freeze``, the real optimizer step, the real exporter as a subprocess, the real
``decode_lattice.load_selector_checkpoint``, the real ``_sample_draft_tokens``, and the real
``decode_lattice.slot_err_logit`` + ``keep_repair_log_probs``.  It needs no GPU and no data.

The final assertion is the one that matters: the native walk and the external ``latrepair`` walk pick
the SAME token at every slot, using a selector that was reloaded from disk through the exporter.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

SPECFORGE = pathlib.Path(__file__).resolve().parents[1]
ROOT = SPECFORGE.parent
sys.path.insert(0, str(SPECFORGE))

from safetensors.torch import save_file  # noqa: E402
from transformers import Qwen3Config  # noqa: E402

from specforge.modeling.draft.pspr_cascade import PSPRCascadeDraftModel  # noqa: E402
from specforge.modeling.draft.pspr_cloze import PSPRClozeDraftModel  # noqa: E402
from specforge.training.model_loading import warm_start_draft_model  # noqa: E402

FAILURES: list[str] = []
ANCHOR_RHO = 3.0
BLOCK = 5
TOP_K = 4
VOCAB = 32
HIDDEN = 16


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def draft_config_dict(architecture: str, **dflash_overrides) -> dict:
    dflash = dict(
        mask_token_id=5,
        target_layer_ids=[0],
        selector_top_k=TOP_K,
        selector_dim=HIDDEN,
        selector_layers=1,
        selector_heads=2,
        selector_delta_hidden=HIDDEN,
        selector_max_slots=8,
        selector_dropout=0.0,
        selector_direct_hidden=True,
        selector_decision_mode="keep_repair",
        selector_compute_dtype="float32",
        selector_state_dim=HIDDEN,
        selector_use_state=True,
        selector_train_scope="err_only",
        selector_err_use_state=False,
        selector_keep_repair_margin=0.0,
        freeze_backbone=True,
        selector_bidirectional=True,
        selector_err_score_detach=True,
        selector_err_score_features="margins_summary",
        selector_err_score_enabled=True,
        selector_err_score_anchor_rho=ANCHOR_RHO,
    )
    dflash.update(dflash_overrides)
    return dict(
        architectures=[architecture],
        block_size=BLOCK,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_target_layers=4,
        head_dim=4,
        max_position_embeddings=64,
        vocab_size=VOCAB,
        tie_word_embeddings=False,
        dflash_config=dflash,
    )


def build(architecture: str, **dflash_overrides):
    payload = draft_config_dict(architecture, **dflash_overrides)
    config = Qwen3Config(**{k: v for k, v in payload.items() if k != "dflash_config"})
    config._attn_implementation = "eager"
    config.dflash_config = payload["dflash_config"]
    torch.manual_seed(0)
    cls = {
        "PSPRClozeDraftModel": PSPRClozeDraftModel,
        "PSPRCascadeDraftModel": PSPRCascadeDraftModel,
    }[architecture]
    return cls(config), payload


class _StubTarget(nn.Module):
    """The two attributes ``_sample_draft_tokens`` reaches for on the target model."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        torch.manual_seed(13)
        nn.init.normal_(self.model.embed_tokens.weight, std=0.5)
        nn.init.normal_(self.lm_head.weight, std=0.5)


def main() -> None:  # noqa: C901
    work = pathlib.Path(tempfile.mkdtemp(prefix="cascade_smoke_"))
    target = _StubTarget().eval()

    # ---- 1: a "champion" cloze checkpoint on disk ------------------------------------------------
    champion, cloze_cfg = build("PSPRClozeDraftModel")
    with torch.no_grad():
        # Make the mature detector visibly nonzero so a step-0 gate that accidentally used it would
        # differ from the anchor, rather than both being near zero by luck.
        champion.candidate_selector.err_head[-1].weight.normal_(std=0.5)
        champion.candidate_selector.err_head[-1].bias.fill_(0.4)
    champion_dir = work / "champion" / "run-step1"
    champion_dir.mkdir(parents=True)
    torch.save(
        {"draft_state_dict": champion.state_dict(), "strategy": "online"},
        champion_dir / "training_state.pt",
    )

    # ---- 2: warm start a cascade model from it, through the real loader --------------------------
    cascade, cascade_cfg = build("PSPRCascadeDraftModel")
    report = warm_start_draft_model(
        cascade,
        str(champion_dir / "training_state.pt"),
        draft_config=cascade.config,
        strategy="online",
    )
    shared = set(champion.state_dict()) & set(cascade.state_dict())
    drift = max(
        float((cascade.state_dict()[k].float() - champion.state_dict()[k].float()).abs().max())
        for k in shared
    )
    check(
        "warm start loaded every champion tensor bit-for-bit and only the cascade head is fresh",
        drift == 0.0 and set(report.missing_keys) == set(cascade.state_dict()) - shared,
        f"max|d| on {len(shared)} shared tensors = {drift:.3e}, "
        f"fresh = {sorted({k.removeprefix('candidate_selector.').split('.')[0] for k in report.missing_keys})}",
    )

    # ---- 3: one real optimizer step on the gate only ---------------------------------------------
    cascade.bind_target_decoder(target.model.embed_tokens)
    frozen = cascade.apply_backbone_freeze()
    ranker_before = {
        k: v.detach().clone()
        for k, v in cascade.state_dict().items()
        if k.startswith("candidate_selector.delta_") or k.startswith("model.")
    }
    trainable = [p for p in cascade.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.5)
    torch.manual_seed(4)
    batch = dict(
        candidate_ids=torch.randint(0, VOCAB, (2, BLOCK - 1, TOP_K)),
        unary_logits=torch.randn(2, BLOCK - 1, TOP_K).log_softmax(-1),
        hidden_states=torch.randn(2, BLOCK - 1, HIDDEN),
        predecessor_ids=torch.randint(0, VOCAB, (2, BLOCK - 1)),
    )
    cascade.train()
    _, err = cascade.candidate_selector.score_candidates(**batch, return_err=True)
    labels = (torch.arange(err.numel(), device=err.device) % 2).float().reshape(err.shape)
    loss = F.binary_cross_entropy_with_logits(err, labels)
    loss.backward()
    grad_norm = sum(float(p.grad.norm()) for p in trainable if p.grad is not None)
    optimizer.step()
    cascade.eval()
    ranker_after_drift = max(
        float((cascade.state_dict()[k].float() - v.float()).abs().max())
        for k, v in ranker_before.items()
    )
    check(
        "one update moved the gate and left the ranker and backbone bit-identical",
        grad_norm > 0.0 and ranker_after_drift == 0.0,
        f"grad norm = {grad_norm:.3e}, ranker/backbone max|d| = {ranker_after_drift:.3e}, "
        f"{frozen} frozen elements",
    )
    check(
        "err_base_gain moved off zero on the first step, so the mature detector is live from step 2",
        float(cascade.candidate_selector.err_base_gain["gain"].detach()) != 0.0,
        f"gain = {float(cascade.candidate_selector.err_base_gain['gain'].detach()):.4e}",
    )

    # ---- 4: checkpoint, then export through the real exporter ------------------------------------
    ckpt_dir = work / "cascade" / "run-step1"
    ckpt_dir.mkdir(parents=True)
    torch.save(
        {
            "draft_state_dict": cascade.state_dict(),
            "strategy": "online",
            "global_step": 1,
            "dflash2_selector_objective": "profitable_repair",
        },
        ckpt_dir / "training_state.pt",
    )
    config_path = work / "cascade-config.json"
    config_path.write_text(json.dumps(cascade_cfg, indent=2))
    cloze_config_path = work / "cloze-config.json"
    cloze_config_path.write_text(json.dumps(cloze_cfg, indent=2))

    template = work / "template"
    template.mkdir()
    (template / "config.json").write_text(json.dumps(cascade_cfg, indent=2))
    reference = work / "reference"
    reference.mkdir()
    save_file(
        {
            k: v.detach().cpu().clone()
            for k, v in cascade.state_dict().items()
            if not k.startswith("candidate_selector.")
        },
        str(reference / "model.safetensors"),
    )

    out = work / "export"
    exporter = [
        sys.executable,
        str(SPECFORGE / "scripts/export_pspr_cascade_for_decode.py"),
        "--checkpoint", str(ckpt_dir / "training_state.pt"),
        "--draft-config", str(config_path),
        "--backbone-template", str(template),
        "--out", str(out),
        "--verify-frozen", str(reference),
    ]
    proc = subprocess.run(exporter, capture_output=True, text=True)
    check(
        "the exporter runs, verifies the frozen backbone, and round-trips strictly",
        proc.returncode == 0
        and "verify-roundtrip" in proc.stdout
        and "verify-frozen" in proc.stdout,
        (proc.stdout + proc.stderr).strip().splitlines()[-1] if proc.returncode else "",
    )

    # The same checkpoint with the CLOZE draft-config must be refused, not silently mislabelled.
    bad = subprocess.run(
        exporter[:5] + [str(cloze_config_path)] + exporter[6:],
        capture_output=True,
        text=True,
    )
    check(
        "exporting with a mismatched architecture in the draft-config is refused up front",
        bad.returncode != 0 and "architectures" in (bad.stdout + bad.stderr),
        (bad.stdout + bad.stderr).strip().splitlines()[-1][:120],
    )

    # ---- 5: strict reload through the external decoder's own loader ------------------------------
    spec = importlib.util.spec_from_file_location(
        "decode_lattice", ROOT / "TAPS-SP/scripts/decode_lattice.py"
    )
    decode = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "TAPS-SP/scripts"))
    spec.loader.exec_module(decode)
    reloaded, payload = decode.load_selector_checkpoint(
        str(out / "selector.pt"), device="cpu"
    )
    reloaded.bind_target_embedding(target.model.embed_tokens)
    reloaded.eval()
    live = cascade.candidate_selector
    reload_drift = max(
        float((reloaded.state_dict()[k].float() - live.state_dict()[k].float()).abs().max())
        for k in live.state_dict()
        if k != "target_embedding"
    )
    check(
        "the decoder's loader rebuilds the selector bit-for-bit, including the cascade knobs",
        reload_drift == 0.0
        and reloaded.err_score_features == live.err_score_features
        and reloaded.err_score_anchor_rho == live.err_score_anchor_rho
        and reloaded.err_score_enabled == live.err_score_enabled,
        f"max|d| = {reload_drift:.3e}, features={reloaded.err_score_features!r}, "
        f"anchor_rho={reloaded.err_score_anchor_rho}, enabled={reloaded.err_score_enabled}",
    )
    check(
        "the artefact records the training objective and the serving policy",
        payload["training_policy"]["selector_objective"] == "profitable_repair"
        and payload["decode_policy"]["selector_decision_mode"] == "keep_repair"
        and payload["decode_policy"]["selector_keep_repair_margin"] == 0.0,
        f"{payload['training_policy']['selector_objective']!r}, "
        f"{payload['decode_policy']['selector_decision_mode']!r}",
    )

    # ---- 6: one native block ---------------------------------------------------------------------
    torch.manual_seed(9)
    draft_hidden = torch.randn(1, BLOCK, HIDDEN)
    block_output_ids = torch.randint(0, VOCAB, (1, BLOCK))
    with torch.no_grad():
        native_path = cascade._sample_draft_tokens(target, draft_hidden, block_output_ids)
    check(
        "native serving completes a keep_repair block with the cascade gate",
        native_path.shape[-1] == BLOCK - 1,
        f"path = {native_path.tolist()}",
    )

    # ---- 7: one external latrepair block, on the RELOADED selector ------------------------------
    # Same lattice, same anchor, same arithmetic as decode_lattice's `latrepair` branch, driven by the
    # module that came back off disk.
    hidden = draft_hidden[:, -BLOCK + 1 :, :]
    with torch.no_grad():
        unary, cand_ids, scalars = reloaded.extract_lattice(target.lm_head(hidden))
        embedding = reloaded._embedding()
        cand_emb = F.embedding(cand_ids, embedding)
        z = reloaded.cloze_states(
            hidden, cand_ids, block_output_ids[:, 0], unary, scalars
        )
        prev = block_output_ids[:, 0]
        gru_hidden = None
        external_path = []
        for position in range(cand_ids.shape[1]):
            step, gru_hidden = reloaded.gru(
                F.embedding(prev, embedding)
                .unsqueeze(1)
                .to(reloaded.gru.weight_ih_l0.dtype),
                gru_hidden,
            )
            state = step[:, 0]
            sc = reloaded.score(
                z[:, position], hidden[:, position], cand_emb[:, position],
                unary[:, position], state,
            )[0]
            el = decode.slot_err_logit(
                reloaded, z[0], state.unsqueeze(0), unary[0], scalars[0],
                cand_emb[0], hidden, position, scores=sc,
            ).reshape(1)
            action = decode.keep_repair_log_probs(reloaded, sc.unsqueeze(0), el)[0]
            token = int(cand_ids[0, position, int(action.argmax())])
            external_path.append(token)
            prev = torch.tensor([token])
    check(
        "external latrepair on the reloaded selector picks the SAME token at every slot as native",
        external_path == native_path[0].tolist(),
        f"native={native_path[0].tolist()}, external={external_path}",
    )

    print()
    print(f"workdir: {work}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        raise SystemExit(1)
    print("end-to-end smoke passed")


if __name__ == "__main__":
    main()
