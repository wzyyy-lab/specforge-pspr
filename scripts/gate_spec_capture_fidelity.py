"""G0.3: prove the ported spec-capture patch produces the right tensors, not just well-formed ones.

The patch was written against sglang 0.5.18 and 7 of its 38 hunks had to be transplanted by hand onto
the installed 0.5.15. A transplant that lands in the wrong place does not crash: a wrong layer
choice, a wrong ``hidden_states`` index offset, a wrong concat order, or a prefill silently split
into chunks all yield a perfectly well-formed ``[1, seq, 12800]`` bf16 tensor. SpecForge's own
runtime checks (``specforge/inference/capture.py:97-120``) compare recorded aux-layer ids and width,
which catches none of those.

So compare values. ``scripts/dump_hidden_states_hf.py`` is the independently-written HF reference that
produced the existing 84G offline cache, and the offline and online paths are contractually required
to be interchangeable (same ``resolve_dflash_capture_layers``, same collator, same ``offset=1``
semantics). Feeding both the same ``input_ids`` must therefore give the same tensor.

Run (needs 2 free GPUs: one for the capture server, one for the HF reference)::

    PYTHONPATH=. python scripts/gate_spec_capture_fidelity.py \
        --sglang-path /home/wangzhuoyu/sglang-patched \
        --target-model /kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B \
        --draft-config configs/qwen3-4b-pspr-joint.json \
        --data cache/dataset/perfectblend_200k.jsonl \
        --num-samples 4 --server-device 0 --reference-device 1
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MOONCAKE_RPC_PORT = 50151
MOONCAKE_METADATA_PORT = 8181
MOONCAKE_METRICS_PORT = 9191


def post_json(url: str, body: dict, timeout: float) -> object:
    payload = json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def wait_ready(url: str, timeout_s: float, what: str, *, tcp_port: int | None = None) -> None:
    """Poll until ready, using launch_plan._http_ready's exact criteria.

    Mooncake's metadata endpoint answers a read-only probe for a missing key with 4xx, which means
    the service is up; only 5xx means listening-but-unhealthy. When ``tcp_port`` is given the RPC
    port must also accept a connection, matching ``_readiness_satisfied``.
    """
    import socket

    started = time.time()
    deadline = started + timeout_s
    last = ""
    while time.time() < deadline:
        http_ok = False
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                status = getattr(response, "status", 200)
                http_ok = tcp_port is not None or 200 <= status < 300
        except urllib.error.HTTPError as exc:
            http_ok = tcp_port is not None and 400 <= exc.code < 500
            last = f"HTTP {exc.code}"
        except (OSError, urllib.error.URLError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        if http_ok and tcp_port is not None:
            try:
                with socket.create_connection(("127.0.0.1", tcp_port), timeout=2.0):
                    pass
            except OSError as exc:
                http_ok = False
                last = f"rpc port {tcp_port}: {exc}"
        if http_ok:
            print(f"  {what} ready after {time.time() - started:.0f}s", flush=True)
            return
        time.sleep(2)
    raise SystemExit(f"{what} not ready within {timeout_s}s (last: {last})")


def mooncake_env() -> dict:
    return {
        "MOONCAKE_METADATA_SERVER": f"http://127.0.0.1:{MOONCAKE_METADATA_PORT}/metadata",
        "MOONCAKE_MASTER_SERVER_ADDR": f"127.0.0.1:{MOONCAKE_RPC_PORT}",
        "MOONCAKE_LOCAL_HOSTNAME": "127.0.0.1",
        "MOONCAKE_PROTOCOL": "tcp",
    }


def prompts(path: Path, tokenizer, chat_template: str, max_length: int, want: int) -> list:
    """Tokenize a few rows exactly as scripts/dump_hidden_states_hf.py does, so the two paths are
    being compared on identical token sequences and identical supervision spans."""
    import torch

    from specforge.data.loss_mask import has_consecutive_supervised_tokens
    from specforge.data.preprocessing import preprocess_conversations
    from specforge.data.template import TEMPLATE_REGISTRY

    template = TEMPLATE_REGISTRY.get(chat_template)
    rows = []
    with path.open() as handle:
        for line in handle:
            if len(rows) >= want:
                break
            record = json.loads(line)
            result = preprocess_conversations(
                tokenizer, [record["conversations"]], template, max_length=max_length
            )
            if not result["input_ids"]:
                continue
            input_ids = result["input_ids"][0][0].to(torch.long)
            loss_mask = result["loss_mask"][0][0].to(torch.long)
            if not has_consecutive_supervised_tokens(loss_mask):
                continue
            if input_ids.shape[0] < 16:
                continue
            rows.append(
                {
                    "id": record["id"],
                    "input_ids": [int(v) for v in input_ids.tolist()],
                    "loss_mask": [int(v) for v in loss_mask.tolist()],
                }
            )
    if len(rows) < want:
        raise SystemExit(f"only {len(rows)} usable prompts in {path}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang-path", required=True, help="patched sglang parent dir")
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--draft-config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--server-device", default="0")
    ap.add_argument("--reference-device", default="1")
    ap.add_argument("--port", type=int, default=31000)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--chat-template", default="qwen")
    ap.add_argument("--mem-fraction-static", type=float, default=0.5)
    ap.add_argument(
        "--attention-backend",
        default="flashinfer",
        help="must match cfg.model.sglang_attention_backend for the run being gated",
    )
    ap.add_argument("--startup-timeout", type=float, default=1800.0)
    ap.add_argument("--log-dir", default="outputs/gate_spec_capture")
    ap.add_argument(
        "--use-running",
        action="store_true",
        help="attach to an already-running mooncake + capture server on the configured ports "
        "instead of launching them; the server takes ~340s to load weights off NFS, so reusing one "
        "makes the check re-runnable",
    )
    args = ap.parse_args()

    import torch

    draft_config = json.loads(Path(args.draft_config).read_text())
    layer_ids = list(draft_config["dflash_config"]["target_layer_ids"])
    hidden_size = int(draft_config["hidden_size"])
    expected_width = len(layer_ids) * hidden_size
    print(f"contract: layers={layer_ids} hidden={hidden_size} width={expected_width}")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    procs: list[subprocess.Popen] = []

    def spawn(name: str, argv: list, env_extra: dict) -> subprocess.Popen:
        env = {**os.environ, **env_extra}
        handle = (log_dir / f"{name}.log").open("w")
        print(f"  launching {name} -> {log_dir / f'{name}.log'}")
        proc = subprocess.Popen(
            argv, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
        )
        procs.append(proc)
        return proc

    try:
        if args.use_running:
            print("attaching to already-running mooncake + capture server")
            wait_ready(
                f"http://127.0.0.1:{MOONCAKE_METADATA_PORT}/metadata?key=gate-health",
                30.0,
                "mooncake",
                tcp_port=MOONCAKE_RPC_PORT,
            )
            wait_ready(f"http://127.0.0.1:{args.port}/health", 60.0, "capture server")
        else:
            # --- phase 0: mooncake master (the sink writes captured tensors into it) ------------
            spawn(
                "mooncake",
                [
                    "mooncake_master",
                    "--enable_http_metadata_server=true",
                    "--http_metadata_server_host=127.0.0.1",
                    f"--rpc_port={MOONCAKE_RPC_PORT}",
                    f"--http_metadata_server_port={MOONCAKE_METADATA_PORT}",
                    f"--metrics_port={MOONCAKE_METRICS_PORT}",
                    "--default_kv_lease_ttl=500",
                ],
                {"CUDA_VISIBLE_DEVICES": ""},
            )
            wait_ready(
                f"http://127.0.0.1:{MOONCAKE_METADATA_PORT}/metadata?key=gate-health",
                120.0,
                "mooncake",
                tcp_port=MOONCAKE_RPC_PORT,
            )

            # --- phase 1: patched capture server ------------------------------------------------
            # argv mirrors what `specforge train --plan` resolves for this recipe, field for field:
            #   -m sglang.launch_server --model-path ... --dtype bfloat16 --skip-tokenizer-init
            #   --tp-size 1 --chunked-prefill-size -1 --enable-spec-capture --spec-capture-method
            #   dflash --spec-capture-aux-layer-ids 1 9 17 25 33 --host 127.0.0.1 --port ...
            #   --attention-backend flashinfer --mem-fraction-static 0.5 --disable-radix-cache
            #   --context-length 3200 --ep-size 1
            # The attention backend is NOT cosmetic here: it decides which kernels produce the
            # hidden states being compared, so validating on a different backend than production
            # runs would leave the numbers this gate reports unattached to the training run.
            spawn(
                "capture-server",
                [
                    sys.executable,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    args.target_model,
                    "--dtype",
                    "bfloat16",
                    "--skip-tokenizer-init",
                    "--tp-size",
                    "1",
                    # Capture needs a single-pass prefill; chunking would drop all but one chunk.
                    "--chunked-prefill-size",
                    "-1",
                    "--enable-spec-capture",
                    "--spec-capture-method",
                    "dflash",
                    "--spec-capture-aux-layer-ids",
                    *[str(layer) for layer in layer_ids],
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                    "--attention-backend",
                    args.attention_backend,
                    "--mem-fraction-static",
                    str(args.mem_fraction_static),
                    # Prefix reuse would serve a partially cached prefill and capture only the tail.
                    "--disable-radix-cache",
                    "--context-length",
                    str(args.max_length + 128),
                    "--ep-size",
                    "1",
                ],
                {
                    **mooncake_env(),
                    "PYTHONPATH": args.sglang_path,
                    "CUDA_VISIBLE_DEVICES": str(args.server_device),
                    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
                    "MOONCAKE_GLOBAL_SEGMENT_SIZE": str(8 << 30),
                    "MOONCAKE_LOCAL_BUFFER_SIZE": str(2 << 30),
                    "HF_HUB_OFFLINE": "1",
                },
            )
            wait_ready(
                f"http://127.0.0.1:{args.port}/health", args.startup_timeout, "capture server"
            )

        # --- build prompts through SpecForge's own preprocessing --------------------------------
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.target_model)
        rows = prompts(
            Path(args.data), tokenizer, args.chat_template, args.max_length, args.num_samples
        )
        print(f"prompts: {[len(r['input_ids']) for r in rows]} tokens")

        # --- capture request. Field-for-field what SGLangServerCaptureAdapter sends -------------
        # Connection kwargs mirror specforge/training/disaggregated.py:133-147; the store's own
        # defaults cover only sizing/protocol, so the three address fields must be passed in.
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

        env = mooncake_env()
        store = MooncakeFeatureStore(
            store_id="gate",
            setup_kwargs={
                "local_hostname": env["MOONCAKE_LOCAL_HOSTNAME"],
                "metadata_server": env["MOONCAKE_METADATA_SERVER"],
                "master_server_addr": env["MOONCAKE_MASTER_SERVER_ADDR"],
                "protocol": env["MOONCAKE_PROTOCOL"],
                "rdma_devices": "",
                # The client contributes no segment; the capture server allocates the objects.
                "global_segment_size": 0,
                "local_buffer_size": 256 << 20,
            },
        )
        payloads = []
        for index, row in enumerate(rows):
            length = len(row["input_ids"])
            payloads.append(
                {
                    "store_id": store.store_id,
                    "sample_id": f"gate:{index}",
                    "gen": 1,
                    "replace": False,
                    "features": {"aux": "hidden_states"},
                    "passthrough": [
                        {
                            "name": "input_ids",
                            "data": row["input_ids"],
                            "shape": [1, length],
                            "dtype": "int64",
                        },
                        {
                            "name": "loss_mask",
                            "data": row["loss_mask"],
                            "shape": [1, length],
                            "dtype": "int64",
                        },
                    ],
                }
            )
        body = {
            "input_ids": [row["input_ids"] for row in rows],
            # A fresh cache namespace forces a full prefill on every attempt.
            "extra_key": [f"gate-{index}-{time.time_ns()}" for index in range(len(rows))],
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
            "spec_capture": payloads,
        }
        print("posting /generate with max_new_tokens=0 (prefill-only capture) ...")
        response = post_json(f"http://127.0.0.1:{args.port}/generate", body, timeout=600.0)
        if not isinstance(response, list):
            response = [response]

        # --- read the captured tensors back out of Mooncake ------------------------------------
        from specforge.runtime.contracts import FeatureSpec, SampleRef

        captured = {}
        for index, item in enumerate(response):
            spec = (item.get("meta_info") or {}).get("spec_capture")
            if not spec:
                raise SystemExit(
                    f"sample {index}: response carries no meta_info['spec_capture']; the server is "
                    "not running the patched capture path"
                )
            recorded = list(spec.get("aux_layer_ids") or [])
            if recorded != layer_ids:
                raise SystemExit(
                    f"sample {index}: aux_layer_ids {recorded} != contract {layer_ids}"
                )
            features = spec["features"]
            entry = features["hidden_states"]
            shape = tuple(int(v) for v in entry["shape"])
            ref = SampleRef(
                sample_id=spec["sample_id"],
                run_id="gate",
                source_task_id=None,
                feature_store_uri="mooncake://gate",
                feature_keys={"hidden_states": "hidden_states"},
                feature_specs={
                    "hidden_states": FeatureSpec(
                        name="hidden_states", shape=shape, dtype=entry["dtype"]
                    )
                },
                strategy="dflash",
                metadata={"generation": int(spec["gen"])},
            )
            tensors, handle = store.get(ref)
            captured[index] = tensors["hidden_states"].clone()
            store.release(handle)
            print(
                f"  sample {index}: captured {tuple(captured[index].shape)} "
                f"{captured[index].dtype}"
            )

        # --- HF reference on a separate device -------------------------------------------------
        print(f"loading HF reference on cuda:{args.reference_device} ...")
        from transformers import AutoModelForCausalLM

        device = f"cuda:{args.reference_device}"
        target = AutoModelForCausalLM.from_pretrained(
            args.target_model, dtype=torch.bfloat16
        ).to(device)
        target.eval()

        failures = []
        for index, row in enumerate(rows):
            input_ids = torch.tensor(row["input_ids"], device=device).unsqueeze(0)
            with torch.no_grad():
                out = target(input_ids, output_hidden_states=True, use_cache=False)
            # offset=1: hidden_states[0] is the embedding output, so layer L lives at index L+1.
            reference = torch.cat(
                [out.hidden_states[layer + 1][0] for layer in layer_ids], dim=-1
            ).to(torch.bfloat16)

            got = captured[index].squeeze(0).to(device, torch.bfloat16)
            if got.shape != reference.shape:
                failures.append(f"sample {index}: shape {tuple(got.shape)} != {tuple(reference.shape)}")
                continue
            delta = (got.float() - reference.float()).abs()
            scale = reference.float().abs().mean().clamp_min(1e-6)
            rel = (delta.mean() / scale).item()
            exact = (got == reference).float().mean().item()
            cos = torch.nn.functional.cosine_similarity(
                got.float().flatten(), reference.float().flatten(), dim=0
            ).item()
            print(
                f"  sample {index}: exact={100 * exact:6.2f}%  max|d|={delta.max():.4f}  "
                f"rel_mean={rel:.2e}  cos={cos:.8f}"
            )
            # bf16 has ~3 decimal digits, and SGLang's fused kernels reassociate sums differently
            # from HF's eager path, so bitwise equality is not the bar. These two are the gross-
            # breakage floor only: a token offset lands at cos 0.03-0.35 and a permuted concat order
            # at 0.04 per slot, both caught here, but a wrong LAYER does not move these numbers
            # meaningfully -- the per-slot ranking check below is what catches that.
            if cos < 0.999:
                failures.append(f"sample {index}: cos={cos:.6f} < 0.999")
            if rel > 0.05:
                failures.append(f"sample {index}: rel_mean={rel:.3e} > 0.05")

            # Per-slot localisation AND layer identification. An absolute cos threshold cannot do the
            # second job. In a residual stream adjacent layers differ by one block's update, and
            # measured on this model cos(layer 9, layer 8) = 0.999923 and cos(layer 17, layer 16) =
            # 0.999927 -- both above any bar loose enough to tolerate bf16 kernel reassociation, which
            # by itself costs only 1 - 0.999995. So an off-by-one layer in the transplant would clear
            # a fixed threshold. What separates them is the ranking: the captured slot must be closer
            # to the layer it claims than to either neighbour. Self-calibrating, no magic constant.
            for slot, layer in enumerate(layer_ids):
                lo, hi = slot * hidden_size, (slot + 1) * hidden_size
                block = got[:, lo:hi].float().flatten()

                def against(hidden_index: int, block=block) -> float:
                    other = out.hidden_states[hidden_index][0].to(torch.bfloat16)
                    return torch.nn.functional.cosine_similarity(
                        block, other.float().flatten(), dim=0
                    ).item()

                block_cos = against(layer + 1)
                neighbours = {
                    offset: against(layer + 1 + offset)
                    for offset in (-1, 1)
                    if 0 <= layer + 1 + offset < len(out.hidden_states)
                }
                nearest = max(neighbours, key=neighbours.get)
                best_wrong = neighbours[nearest]
                print(
                    f"    slot {slot} (layer {layer:2d}): cos={block_cos:.8f}  "
                    f"nearest other layer {layer + nearest:2d} cos={best_wrong:.8f}  "
                    f"margin={block_cos - best_wrong:+.2e}"
                )
                if block_cos < 0.999:
                    failures.append(
                        f"sample {index}: concat slot {slot} (layer {layer}) cos={block_cos:.6f}"
                    )
                if block_cos <= best_wrong:
                    failures.append(
                        f"sample {index}: concat slot {slot} is at least as close to layer "
                        f"{layer + nearest} (cos={best_wrong:.8f}) as to the layer {layer} it claims "
                        f"(cos={block_cos:.8f}): the transplant records the wrong layer"
                    )

        print()
        if failures:
            print("G0.3 FAIL:")
            for line in failures:
                print(f"  - {line}")
            return 1
        print(
            f"G0.3 PASS: {len(rows)} samples, online capture matches the HF reference on all "
            f"{len(layer_ids)} concat slots"
        )
        return 0
    finally:
        for proc in reversed(procs):
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.time() + 30
        for proc in procs:
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.5)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
