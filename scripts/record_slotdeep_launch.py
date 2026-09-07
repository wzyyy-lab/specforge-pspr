"""Save launch-time resolved configuration, source snapshots and input hashes.

No training is started here. Outputs are experiment provenance artifacts;
existing directories are never overwritten. No secret environment is dumped.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import socket
import zipfile

from specforge.config import load_config
from specforge.application import resolve_run
from specforge.launch_plan import build_launch_plan

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--override", action="append", default=[])
    args = p.parse_args()
    cfg = resolve_run(load_config(args.config, args.override))
    plan = build_launch_plan(cfg.config, algorithm=cfg.algorithm,
                             config_path=args.config, overrides=args.override, requested_role="both")
    out = Path(args.out)
    resolved = cfg.config.model_dump(mode="json")
    if Path(resolved["output_dir"]).exists() and not resolved["training"]["resume_from"]:
        raise FileExistsError(f'refusing a fresh launch into existing output: {resolved["output_dir"]}')
    out.mkdir(parents=True, exist_ok=False)
    draft_path = Path(resolved["model"]["draft_model_config"])
    draft = json.loads(draft_path.read_text())
    dataset = Path(resolved["data"]["train_data_path"])
    target = Path(resolved["model"]["target_model_path"])
    backbone = Path(resolved["model"]["draft_checkpoint_path"])
    sources = sorted((ROOT / "specforge").rglob("*.py"))
    sources += sorted((ROOT / "scripts").glob("*.py"))
    sources += sorted((ROOT / "tests/test_modeling").glob("test_pspr*.py"))
    sources += [ROOT / "tests/test_runtime/test_checkpoint_resume.py",
                ROOT / "tests/test_runtime/test_model_loading.py"]
    sources += [ROOT.parent / "TAPS-SP/scripts/decode_lattice.py", Path(args.config), draft_path]
    sources += [ROOT / "scripts/run_train_slotdeep_smoke.sh"]
    hashes = {}
    with zipfile.ZipFile(out / "source_snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            path = path.resolve()
            hashes[str(path)] = sha(path)
            archive.write(path, path.relative_to(ROOT.parent))
    inputs = [dataset]
    if backbone.is_file():
        # Fine-tuning may initialize from training_state.pt, not a directory.
        # globbing it silently records zero checkpoint inputs.
        inputs.append(backbone)
    for folder in (target, backbone):
        inputs += sorted(folder.glob("*.safetensors"))
        inputs += sorted(folder.glob("*.json"))
        inputs += sorted(folder.glob("*.py"))
    input_hashes = {}
    for path in inputs:
        path = path.resolve()
        input_hashes[str(path)] = dict(sha256=sha(path), bytes=path.stat().st_size)
        print("HASHED_INPUT", path.name, path.stat().st_size, flush=True)
    import sglang
    import sglang.srt.spec_capture_sink as capture
    command = ["python", "-u", "-m", "specforge.cli", "train", "-c", args.config, *args.override]
    environment = {name: os.environ.get(name) for name in (
        "PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES",
        "no_proxy", "NO_PROXY", "SLOTDEEP_ENV_SPEC",
    )}
    if environment["SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"] != "1":
        raise ValueError("This host's validated SGLang 0.5.15 capture invocation requires "
                         "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1; see pspr-sglang-0515.md")
    env_spec = Path(os.environ.get("SLOTDEEP_ENV_SPEC", "/home/wangzhuoyu/pspr-sglang-0515-spec.json"))
    spec_content = json.loads(env_spec.read_text())
    spec_hash = hashlib.sha256(json.dumps(spec_content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]
    # Record the import actually used, not just system distribution metadata.
    # This catches image drift even when an old declarative spec file is unchanged.
    import sgl_kernel
    runtime_kernel = dict(version=sgl_kernel.__version__, source=sgl_kernel.__file__)
    try:
        runtime_kernel["distribution_version"] = importlib.metadata.version("sglang-kernel")
    except importlib.metadata.PackageNotFoundError:
        runtime_kernel["distribution_version"] = None
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(),
        cwd=str(ROOT), command=command, pythonpath=os.environ.get("PYTHONPATH"), environment=environment,
        run_id=resolved["run_id"], source_hashes=hashes, input_hashes=input_hashes,
        package_versions={k: importlib.metadata.version(k) for k in ("torch", "transformers", "safetensors")},
        sglang_version=sglang.__version__, capture_source=capture.__file__,
        runtime_kernel=runtime_kernel,
        capture_source_sha256=sha(Path(capture.__file__)),
        environment_spec=str(env_spec), environment_spec_hash=spec_hash,
        semantic_note="Launch-time provenance, not an experiment quality certification.",
    )
    for filename, payload in (("resolved.json", resolved), ("draft_config.json", draft), ("manifest.json", manifest)):
        (out / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (out / "launch_plan.json").write_text(plan.render() + "\n")
    env_prefix = " ".join(f"{name}={shlex.quote(value)}" for name, value in environment.items() if value is not None)
    (out / "COMMAND.txt").write_text(env_prefix + " " + shlex.join(command) + "\n")
    print("LAUNCH_PROVENANCE_SAVED", out.resolve(), flush=True)


if __name__ == "__main__":
    main()
