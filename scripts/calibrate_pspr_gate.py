"""Fit a global rho on declared calibration features; never on decode test labels.

Reuses the fixed-BASE risk accounting. The empirical damage budget is not a
population-risk guarantee. Only rho changes; all model weights stay immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.analyze_pspr_memory_s1 import risk_budget_threshold, risk_counts
from scripts.prepare_pspr_holdout import prompt_key

ROOT = Path(__file__).resolve().parents[1]
METHOD = "pspr_global_rho_v1"


def source_path(path):
    path = Path(path)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def validate_capture_identity(provenance, target_model, draft_config):
    """Legacy captures have paths/layers but no capture-time weight hashes.

    Enforce the recorded identities; do not invent a retroactive weight hash.
    """
    a = provenance['arguments']
    if source_path(a['target_model']) != source_path(target_model):
        raise ValueError('capture target model differs from calibration target')
    if source_path(a['draft_config']) != source_path(draft_config):
        raise ValueError('capture draft config differs from calibration config')
    cfg = json.loads(source_path(draft_config).read_text())
    if provenance['layer_ids'] != cfg['dflash_config']['target_layer_ids']:
        raise ValueError('capture layer IDs differ from draft configuration')


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weight_hashes(directory):
    directory = Path(directory)
    if not list(directory.glob("*.safetensors")):
        raise ValueError(f"no model weights: {directory}")
    # Configuration/tokenizer/custom-model sources are load-bearing too.
    paths = sorted({p for pattern in ("*.safetensors", "*.json", "*.py", "*.model", "*.txt")
                    for p in directory.glob(pattern) if p.is_file()})
    return {p.name: sha(p) for p in paths}


def fit_threshold(kind, margin, slot, max_destroy_rate, log_guard=1e-4):
    """Only base-right labels determine the threshold, never repair labels."""
    if not 0 <= max_destroy_rate < 1 or not math.isfinite(max_destroy_rate):
        raise ValueError("destroy rate must be finite and in [0,1)")
    if not math.isfinite(log_guard) or log_guard < 1e-4:
        raise ValueError("need a finite log-boundary guard >=1e-4")
    native = risk_counts(kind, margin, slot, math.log(3.0))
    negatives = int((kind == 0).sum())
    if negatives == 0:
        raise ValueError("no correct-prefix calibration negatives")
    budget = int(math.floor(max_destroy_rate * negatives))
    boundary = risk_budget_threshold(margin, kind, budget)
    threshold = boundary + log_guard
    if not -20 < threshold < 20:
        raise ValueError("calibrated threshold outside supported finite range")
    rho = math.exp(threshold)
    counts = risk_counts(kind, margin, slot, math.log(rho))
    assert counts["destroyed"] <= budget
    return dict(rho=rho, log_margin_threshold=math.log(rho), boundary=boundary,
                boundary_log_guard=log_guard, damage_budget=budget,
                base_right_n=negatives, max_destroy_rate=max_destroy_rate,
                counts=counts, native_rho3_counts=native)


def calibration_population(prompts, features, target_model, draft_config):
    rows = [json.loads(line) for line in Path(prompts).read_text().splitlines()]
    if len(rows) != 512 or len({r["id"] for r in rows}) != 512:
        raise ValueError("this predeclared calibration cohort must have 512 unique prompts")
    keys = [prompt_key(r["turns"][0]) for r in rows]
    if len(set(keys)) != 512:
        raise ValueError("duplicate calibration prompts")
    files, skipped, provenance = {}, [], {}
    for manifest_path in sorted(Path(features).glob("part*/manifest.json")):
        part = manifest_path.parent
        p = json.loads((part / "provenance.json").read_text())
        validate_capture_identity(p, target_model, draft_config)
        if p["prompts_sha256"] != sha(prompts) or p["arguments"]["split"] != "calibration":
            raise ValueError("features are not from the declared calibration prompts")
        if p["arguments"]["max_prompt_tokens"] != 2816 or p["arguments"]["max_new_tokens"] != 256:
            raise ValueError("feature generation budgets differ")
        m = json.loads(manifest_path.read_text())
        assert len(m) == p["kept"]
        for record in m:
            index = record["prompt_index"]
            row = rows[index]
            assert record["id"] == row["id"] and record["source"] == row["source"]
            assert record["prompt_sha256"] == hashlib.sha256(row["turns"][0].encode()).hexdigest()
            path = (part / record["file"]).resolve()
            assert str(path) not in files and path.is_file()
            files[str(path)] = dict(index=index, sha256=sha(path))
        skipped += p["skipped"]
        provenance[str(manifest_path.resolve())] = sha(manifest_path)
        provenance[str((part / "provenance.json").resolve())] = sha(part / "provenance.json")
    kept = [r["index"] for r in files.values()]
    excluded = [r["index"] for r in skipped]
    assert len(kept) == len(set(kept)) and len(excluded) == len(set(excluded))
    assert set(kept).isdisjoint(excluded) and set(kept) | set(excluded) == set(range(512))
    assert all(r["reason"] in {"prompt_token_limit", "short_generation"} for r in skipped)
    if len(kept) != 511:
        raise ValueError("predeclared cohort expects 511 retained features")
    return dict(prompts_sha256=sha(prompts), normalized_prompt_sha256=sorted(keys),
                feature_files=files, feature_provenance_sha256=provenance,
                retained=len(kept), skipped=skipped, features_path=str(source_path(features)),
                target_model=str(source_path(target_model)), draft_config=str(source_path(draft_config)),
                draft_config_sha256=sha(source_path(draft_config)),
                capture_identity_limit='Legacy captures record target/config paths and layers, not capture-time weight hashes; current hashes do not retroactively prove historical weight identity')


def load_decisions(path, population):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    a = meta["arguments"]
    assert source_path(a['dump_decisions']) == path.resolve(), 'diagnostic NPZ path association differs'
    assert source_path(a['json_output']) == path.with_suffix('.json').resolve(), 'diagnostic JSON path association differs'
    assert a['per_sample_diagnostics'] is True
    assert meta["gates"] and all(meta["gates"].values())
    assert a["include_base_decisions"] and a["preserve_selector_fp32"]
    assert a["gate_rho"] == 3 and a["gate_tau"] == a["gate_theta"] == 0
    assert a["rho_mode"] == "global" and not a["rho_values"]
    assert a["num_anchors"] == 512 and a["max_length"] == 3072 and a["seed"] == 2026
    assert a['num_samples'] == 512 and a['chunk_blocks'] == 64 and a['attention_backend'] == 'flex_attention'
    for name in ('target_model', 'draft_config'):
        if str(source_path(a[name])) != population[name]:
            raise ValueError(f'diagnostic {name} differs from calibration capture identity')
    assert str(source_path(a['features'])) == population['features_path']
    assert meta['input_provenance']['draft_config_sha256'] == population['draft_config_sha256']
    assert meta["input_provenance"]["diagnostic_source_sha256"] == sha(ROOT / "scripts/measure_ranker_teacher_forced.py")
    rows = meta["first_base_error_per_sample"]
    expected = population["feature_files"]
    assert len(rows) == len(expected)
    assert [r["sample_index"] for r in rows] == list(range(len(rows)))
    order = [str(Path(r["feature_file"]).resolve()) for r in rows]
    assert order == sorted(expected)
    for row, key in zip(rows, order):
        assert row["feature_sha256"] == expected[key]["sha256"]
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved["BASE_" + k].copy() for k in ("kind", "margin", "slot", "sample")}
        assert [str(Path(str(p)).resolve()) for p in saved["files"]] == order
    assert np.array_equal(np.unique(arrays["sample"]), np.arange(len(rows)))
    assert len(arrays['sample']) == len(arrays['kind'])
    c = meta["populations"]["BASE"]["counters"]
    assert (arrays["kind"] == 0).sum() == c["base_right_n"]
    assert (arrays["kind"] != 0).sum() == c["base_wrong_n"]
    assert np.isin(arrays["kind"], [1, 2]).sum() == c["fixable_n"]
    observed = risk_counts(arrays["kind"], arrays["margin"], arrays["slot"], math.log(3))
    assert observed["repaired"] == c["fix_recovered"] and observed["destroyed"] == c["base_right_destroyed"]
    return arrays, meta


def model_identity(export, diagnostic_meta, target_model):
    if source_path(diagnostic_meta['arguments']['target_model']) != source_path(target_model):
        raise ValueError('diagnostic target model differs from exported calibration target')
    import torch
    from safetensors.torch import load_file

    export = Path(export).resolve()
    checkpoint = source_path(diagnostic_meta["arguments"]["checkpoint"])
    checkpoint_sha = sha(checkpoint)
    assert checkpoint_sha == diagnostic_meta["input_provenance"]["checkpoint_sha256"]
    payload = torch.load(export / "selector.pt", map_location="cpu", weights_only=False, mmap=True)
    trained = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    assert trained["global_step"] == payload["global_step"] == 7115
    states = trained["draft_state_dict"]
    selector_states = {k.removeprefix("candidate_selector."): v for k, v in states.items()
                       if k.startswith("candidate_selector.") and not k.endswith("target_embedding")}
    assert set(selector_states) == set(payload["state_dict"])
    assert all(torch.equal(v, payload["state_dict"][k]) for k, v in selector_states.items())
    bb = load_file(str(export / "backbone/model.safetensors"))
    expected_bb = {k: v for k, v in states.items() if not k.startswith("candidate_selector.")}
    assert set(bb) == set(expected_bb) and all(torch.equal(v, bb[k]) for k, v in expected_bb.items())
    p = payload["decode_policy"]
    assert payload["model_type"] == "SlotDeepCorrector" and payload["training_policy"]["verified"]
    assert p["selector_decision_mode"] == "margin_gate" and p["selector_compute_dtype"] == "float32"
    assert p["selector_top_k"] == 16 and p["selector_gate_rho"] == 3
    assert p["selector_gate_tau"] == p["selector_gate_theta"] == 0 and not p["selector_gate_skip_first"]
    assert all(v.dtype == torch.float32 for v in payload["state_dict"].values() if v.is_floating_point())
    return dict(export=str(export), checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                selector_sha256=sha(export / "selector.pt"), backbone_sha256=weight_hashes(export / "backbone"),
                target_model=str(Path(target_model).resolve()), target_sha256=weight_hashes(target_model),
                exported_decode_policy=p, checkpoint_global_step=7115)


def validate_for_evaluation(artifact, arm, export, target_model, prompts, decoder_path):
    """Fail before GPU loading; never silently replace unrelated policy fields."""
    d = json.loads(Path(artifact).read_text())
    if d.get("schema_version") != 1 or d.get("method") != METHOD or arm not in d["models"]:
        raise ValueError("unsupported calibration artifact or missing arm")
    if d.get("score_temperature") != 1.0:
        raise ValueError("calibration requires score_temperature=1")
    m = d["models"][arm]
    export = Path(export)
    if m["selector_sha256"] != sha(export / "selector.pt") or m["backbone_sha256"] != weight_hashes(export / "backbone"):
        raise ValueError("calibration model weights mismatch")
    if m["target_sha256"] != weight_hashes(target_model) or d["reference_decoder_sha256"] != sha(decoder_path):
        raise ValueError("calibration target/decoder identity mismatch")
    rho = m["fit"]["rho"]
    if not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho <= 0:
        raise ValueError("invalid calibrated rho")
    if not math.isclose(math.log(rho), m["fit"]["log_margin_threshold"], abs_tol=1e-12):
        raise ValueError("calibration log/rho mismatch")
    rows = [json.loads(line) for line in Path(prompts).read_text().splitlines()]
    observed = {prompt_key(row["turns"][0]) for row in rows}
    if observed.intersection(d["calibration"]["normalized_prompt_sha256"]):
        raise ValueError("evaluation overlaps calibration prompts")
    policy = dict(m["exported_decode_policy"])
    policy["selector_gate_rho"] = rho
    return policy, dict(method=METHOD, artifact=str(Path(artifact).resolve()), artifact_sha256=sha(artifact),
                        arm=arm, calibration_prompts_sha256=d["calibration"]["prompts_sha256"],
                        independent_prompt_check=True, original_policy=m["exported_decode_policy"],
                        fitted_rho=rho, max_destroy_rate=m["fit"]["max_destroy_rate"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--decisions", action="append", required=True, help="LABEL=NPZ")
    p.add_argument("--export", action="append", required=True, help="LABEL=DIR")
    p.add_argument("--target-model", type=Path, required=True)
    p.add_argument("--max-destroy-rate", type=float, default=0.005)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    exports = dict(item.split("=", 1) for item in args.export)
    decisions = dict(item.split("=", 1) for item in args.decisions)
    assert len(exports) == len(args.export) == len(decisions) == len(args.decisions) == 2
    assert set(exports) == set(decisions)
    population = calibration_population(args.prompts, args.features, args.target_model,
                                        ROOT / 'configs/qwen3-4b-pspr-slotdeep.json')
    models, reference = {}, None
    for label, path in decisions.items():
        arrays, meta = load_decisions(path, population)
        if reference is None:
            reference = arrays
        for key in ("sample", "slot"):
            assert np.array_equal(arrays[key], reference[key])
        for kind in (0, 3):
            assert np.array_equal(arrays["kind"] == kind, reference["kind"] == kind)
        identity = model_identity(exports[label], meta, args.target_model)
        fitted = fit_threshold(arrays["kind"], arrays["margin"], arrays["slot"], args.max_destroy_rate)
        models[label] = dict(**identity, fit=fitted, diagnostic=str(Path(path).resolve()),
                             diagnostic_npz_sha256=sha(path), diagnostic_json_sha256=sha(Path(path).with_suffix(".json")))
        print("CALIBRATED", label, json.dumps(fitted), flush=True)
    decoder = ROOT.parent / "TAPS-SP/scripts/decode_lattice.py"
    output = dict(schema_version=1, method=METHOD, created_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Fixed HF teacher-prefix empirical risk calibration; not a deployment risk guarantee or a native result",
                  calibration=population, models=models, reference_decoder_sha256=sha(decoder),
                  calibration_source_sha256=sha(__file__), score_temperature=1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as sink:
        json.dump(output, sink, indent=2)
        sink.write("\n")
    print("CALIBRATION_FROZEN", args.output, flush=True)


if __name__ == "__main__":
    main()
