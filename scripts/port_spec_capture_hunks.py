#!/usr/bin/env python3
"""Port the 7 spec-capture hunks that `patch --fuzz=3` could not place on sglang 2dd449ce5.

Context: `managed_local` training refuses to start without `sglang.srt.spec_capture_sink`
(`specforge/launch_plan.py:962-971`).  The installed sglang in this environment is an upstream
checkout at 2dd449ce5, which matches neither shipped patch target (v0.5.18, kimi-k3-f8493a4), and
the previously patched tree is gone -- which is the same environment change that made the historical
decode baselines irreproducible.

`scripts/apply_sglang_spec_capture_patch.sh` cannot be used as-is for a second reason: it runs
`git apply -p2` with cwd `<root>/python`, and because that directory sits INSIDE the sglang git
repository, git resolves the patch paths against the repo root, finds them outside cwd, and reports
"Skipped patch ..." for every file while still exiting 0.  The script then prints "applied" and
records a patch that never landed.

So: `patch -p1 --fuzz=3` places 31 of 38 hunks, and this script places the remaining 7 against the
current source.  Every replacement is an exact string match with an assertion, so a future sglang
that has drifted again fails loudly instead of half-patching.

Run: sudo -n python3 scripts/port_spec_capture_hunks.py [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/sgl-workspace/sglang/python/sglang/srt")

EDITS: list[tuple[str, str, str, str]] = [
    # ---------------------------------------------------------------- schedule_batch.py, hunk 2
    # This sglang has no `return_hidden_states_mode` / `get_return_hidden_states_mode`, which is why
    # the hunk's context failed.  The substance is unchanged: a capture request forces
    # `return_hidden_states` on, and three accumulators are attached to the Req.
    (
        "managers/schedule_batch.py",
        "schedule_batch: attach spec_capture accumulators to Req",
        """        self.custom_logit_processor = custom_logit_processor
        self.return_hidden_states = return_hidden_states
""",
        """        self.custom_logit_processor = custom_logit_processor
        # Spec-training capture: piggyback the return_hidden_states path (fires
        # CaptureHiddenMode.FULL); the sink consumes the slices, not the response.
        self.spec_capture = spec_capture
        if spec_capture is not None:
            return_hidden_states = True
        self.return_hidden_states = return_hidden_states
        self.spec_capture_aux: List[torch.Tensor] = []
        self.spec_capture_last_hidden: List[torch.Tensor] = []
        self.spec_capture_result = None  # per-request sink result -> output field
""",
    ),
    # ---------------------------------------------------------------- scheduler.py, all 10 hunks
    # `patch --fuzz=3` placed the `drain_spec_captures()` insertion INSIDE the signature of
    # `add_external_corpus`, a SyntaxError at import, so scheduler.py was reverted to HEAD and every
    # hunk is placed here by exact anchor.  This version has three `copy_to_cpu` sites, not four.
    (
        "managers/scheduler.py",
        "scheduler: init the capture sink and require single-pass prefill",
        """        self.init_batch_result_processor()

        self.is_initializing = False
""",
        """        self.init_batch_result_processor()

        if server_args.enable_spec_capture:
            # Capture needs single-pass prefill; chunking would drop all but the
            # final chunk's hidden rows.
            if server_args.chunked_prefill_size != -1:
                raise ValueError(
                    "--enable-spec-capture requires --chunked-prefill-size -1 "
                    "(single-pass prefill) so captured hidden states cover the "
                    "whole sequence"
                )
            from sglang.srt import spec_capture_sink

            spec_capture_sink.maybe_init_sink(server_args)

        self.is_initializing = False
""",
    ),
    (
        "managers/scheduler.py",
        "scheduler: forward spec_capture from the request onto the Req",
        """                dllm_config=self.dllm_config,
                time_stats=recv_req.time_stats,
                multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
            )
            req.tokenizer = self.tokenizer
""",
        """                dllm_config=self.dllm_config,
                time_stats=recv_req.time_stats,
                multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
                spec_capture=recv_req.spec_capture,
            )
            req.tokenizer = self.tokenizer
""",
    ),
    (
        "managers/scheduler.py",
        "scheduler: add _should_copy_hidden_states_to_cpu",
        """    @contextmanager
    def _forward_isolation(self, batch: ScheduleBatch, *, overlap: bool):
""",
        """    def _should_copy_hidden_states_to_cpu(self, batch: ScheduleBatch) -> bool:
        \"\"\"Avoid redundant multi-GiB capture D2H on non-writer TP ranks.\"\"\"
        return batch.return_hidden_states and (
            self.ps.attn_tp_rank == 0
            or any(req.spec_capture is None for req in batch.reqs)
        )

    @contextmanager
    def _forward_isolation(self, batch: ScheduleBatch, *, overlap: bool):
""",
    ),
    (
        "managers/scheduler.py",
        "scheduler: overlap-stash copy_to_cpu",
        """                            self.future_map.stash(future_indices, stash_payload)
                            batch_result.copy_to_cpu(
                                return_logprob=batch.return_logprob,
                                return_hidden_states=batch.return_hidden_states,
                            )""",
        """                            self.future_map.stash(future_indices, stash_payload)
                            batch_result.copy_to_cpu(
                                return_logprob=batch.return_logprob,
                                return_hidden_states=self._should_copy_hidden_states_to_cpu(
                                    batch
                                ),
                            )""",
    ),
    (
        "managers/scheduler.py",
        "scheduler: non-overlap sync copy_to_cpu",
        """                batch_result.copy_done = self.device_module.Event()
                batch_result.copy_to_cpu(
                    return_logprob=batch.return_logprob,
                    return_hidden_states=batch.return_hidden_states,
                )""",
        """                batch_result.copy_done = self.device_module.Event()
                batch_result.copy_to_cpu(
                    return_logprob=batch.return_logprob,
                    return_hidden_states=self._should_copy_hidden_states_to_cpu(
                        batch
                    ),
                )""",
    ),
    (
        "managers/scheduler.py",
        "scheduler: delay-sample copy_to_cpu",
        """            batch_result.copy_to_cpu(
                return_logprob=self.cur_batch.return_logprob,
                return_hidden_states=self.cur_batch.return_hidden_states,
            )""",
        """            batch_result.copy_to_cpu(
                return_logprob=self.cur_batch.return_logprob,
                return_hidden_states=self._should_copy_hidden_states_to_cpu(
                    self.cur_batch
                ),
            )""",
    ),
    (
        "managers/scheduler.py",
        "scheduler: drain capture transfers before processing any result",
        """        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        self.publish_load_snapshot(force=batch.forward_mode.is_extend())
""",
        """        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        # Complete capture transfers on the scheduler thread before processing
        # any kind of next result (prefill, decode, or idle).  This preserves
        # the output socket's single-thread ownership under mixed traffic.
        self.batch_result_processor.drain_spec_captures()

        self.publish_load_snapshot(force=batch.forward_mode.is_extend())
""",
    ),
    (
        "managers/scheduler.py",
        "scheduler: do not enter the idle sleeper with a capture future outstanding",
        """    def on_idle(self):
        \"\"\"Idle housekeeping: guard, check, metrics, reset, sleep.\"\"\"
        if not self.is_fully_idle():
            return
""",
        """    def on_idle(self):
        \"\"\"Idle housekeeping: guard, check, metrics, reset, sleep.\"\"\"
        # A capture response is intentionally delayed until its background
        # sink transfer completes; finish it on this thread so ownership of
        # the ZeroMQ output socket remains single-threaded.  Do not enter the
        # idle sleeper while a future is outstanding or the producer waiting
        # for that response could deadlock.
        self.batch_result_processor.drain_spec_captures()
        if self.batch_result_processor.has_pending_spec_captures():
            time.sleep(0.001)
            return
        if not self.is_fully_idle():
            return
""",
    ),
    (
        "managers/scheduler.py",
        "scheduler: a pending capture means the engine is not idle",
        """            and (not self.enable_overlap or len(self.result_queue) == 0)
            and self._pp_microbatches_drained()
        )""",
        """            and (not self.enable_overlap or len(self.result_queue) == 0)
            and not self.batch_result_processor.has_pending_spec_captures()
            and self._pp_microbatches_drained()
        )""",
    ),
    # ------------------------------------------------- batch_result_processor.py, hunk 4
    # This sglang's prefill path calls `_append_prefill_hidden_states(req, logits_output,
    # hidden_state_offset)` and derives the span from `len(req.origin_input_ids)`; the patch's
    # version took `capture_hidden_mode`, `extend_input_len` and `store`.  Only the branch is
    # ported: capture requests go to `_append_spec_capture_states`, which needs the explicit span,
    # and `extend_input_len_per_req` plus the loop index `i` are both in scope here
    # (`process_batch_result_prefill`).
    (
        "managers/scheduler_components/batch_result_processor.py",
        "batch_result_processor: branch prefill hidden-state collection to the capture sink",
        """                    if (
                        req.return_hidden_states
                        and logits_output.hidden_states is not None
                    ):
                        hidden_state_offset = self._append_prefill_hidden_states(
                            req=req,
                            logits_output=logits_output,
                            hidden_state_offset=hidden_state_offset,
                        )""",
        """                    if (
                        req.return_hidden_states
                        and logits_output.hidden_states is not None
                    ):
                        if req.spec_capture is not None:
                            assert extend_input_len_per_req is not None
                            hidden_state_offset = self._append_spec_capture_states(
                                req=req,
                                logits_output=logits_output,
                                hidden_state_offset=hidden_state_offset,
                                extend_input_len=extend_input_len_per_req[i],
                            )
                        else:
                            hidden_state_offset = self._append_prefill_hidden_states(
                                req=req,
                                logits_output=logits_output,
                                hidden_state_offset=hidden_state_offset,
                            )""",
    ),
    # ------------------------------------------------- batch_result_processor.py, hunk 6
    # A capture request must not be streamed to the client until the Mooncake transfer has completed,
    # otherwise the response carries no `spec_capture` field.  The prefill path in this sglang passes
    # a third positional argument (`skip_stream_req`) that the patch's version did not have.
    (
        "managers/scheduler_components/batch_result_processor.py",
        "batch_result_processor: hold capture requests out of the prefill stream_output",
        """        self.output_streamer.stream_output(
            batch.reqs, batch.return_logprob, skip_stream_req
        )

        can_run_cuda_graph = result.can_run_cuda_graph""",
        """        pending_spec_captures: List = []
        for req in batch.reqs:
            if req.spec_capture is not None and req.finished():
                pending = self._sink_spec_capture(req)
                if pending is not None:
                    pending_spec_captures.append(pending)
        if pending_spec_captures:
            self._queue_spec_captures(
                pending_spec_captures,
                return_logprob=batch.return_logprob,
            )
            capture_req_ids = {id(item[0]) for item in pending_spec_captures}
            ready_reqs = [
                req
                for req in batch.reqs
                if id(req) not in capture_req_ids and req is not skip_stream_req
            ]
            if ready_reqs:
                self.output_streamer.stream_output(ready_reqs, batch.return_logprob)
        else:
            self.output_streamer.stream_output(
                batch.reqs, batch.return_logprob, skip_stream_req
            )

        can_run_cuda_graph = result.can_run_cuda_graph""",
    ),
    # ------------------------------------------------- output_streamer.py, hunk 4
    # The accumulator already collects `req.spec_capture_result` (an applied hunk); this is the line
    # that puts it on the outgoing BatchStrOutput.
    (
        "managers/scheduler_components/output_streamer.py",
        "output_streamer: emit the spec_capture field",
        """            output_hidden_states=self.output_hidden_states,
            routed_experts=self.routed_experts,""",
        """            output_hidden_states=self.output_hidden_states,
            spec_capture=self.spec_capture or None,
            routed_experts=self.routed_experts,""",
    ),
    # ---------------------------------------------------------------- server_args.py
    # This sglang declares ServerArgs as a plain dataclass (`name: type = default`) and registers
    # 417 hand-written `add_argument` calls; the patch was written against the annotated
    # `A[..., NS("exec.features")]` style, so `patch --fuzz=3` placed the three new fields INSIDE a
    # method body and orphaned an `assert`, which is an IndentationError at import.  The file was
    # reverted and the same three options are declared in the local style instead.
    (
        "server_args.py",
        "server_args: declare the spec-capture fields",
        """    enable_return_hidden_states: bool = False
    enable_return_routed_experts: bool = False
""",
        """    enable_return_hidden_states: bool = False
    # Server-side speculative-training capture: per-request aux/last hidden states go to a Mooncake
    # store in the SpecForge DataFlow layout instead of the response payload, so aux-hidden-state
    # capture needs no speculative draft worker.
    enable_spec_capture: bool = False
    spec_capture_aux_layer_ids: Optional[List[int]] = None
    spec_capture_method: str = "eagle3"
    enable_return_routed_experts: bool = False
""",
    ),
    (
        "server_args.py",
        "server_args: register the spec-capture CLI options",
        """        parser.add_argument(
            "--enable-return-routed-experts",
            action="store_true",
            help="Enable returning routed experts of each layer with responses.",
        )
""",
        """        parser.add_argument(
            "--enable-spec-capture",
            action="store_true",
            help=(
                "Enable server-side speculative-training capture: per-request aux/last hidden "
                "states are written to a Mooncake store (SpecForge DataFlow layout) instead of "
                "the response payload."
            ),
        )
        parser.add_argument(
            "--spec-capture-aux-layer-ids",
            type=int,
            nargs="+",
            default=ServerArgs.spec_capture_aux_layer_ids,
            help=(
                "Target layer ids whose hidden states are captured (concatenated) for "
                "spec-capture requests. Defaults to the model's EAGLE3 default layers when unset."
            ),
        )
        parser.add_argument(
            "--spec-capture-method",
            type=str,
            default=ServerArgs.spec_capture_method,
            choices=["eagle3", "dflash", "dspark"],
            help=(
                "Capture method for --enable-spec-capture. Must match the draft strategy being "
                "trained; they wire capture onto different submodules."
            ),
        )
        parser.add_argument(
            "--enable-return-routed-experts",
            action="store_true",
            help="Enable returning routed experts of each layer with responses.",
        )
""",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report state without writing")
    args = ap.parse_args()

    todo, done, bad = [], [], []
    for rel, label, old, new in EDITS:
        path = ROOT / rel
        text = path.read_text()
        if new in text:
            done.append(label)
        elif text.count(old) == 1:
            todo.append((path, label, old, new))
        else:
            bad.append(f"{label}: {text.count(old)} matches of the anchor in {rel}")

    for label in done:
        print(f"  [already] {label}")
    for label in (t[1] for t in todo):
        print(f"  [pending] {label}")
    for msg in bad:
        print(f"  [FAIL]    {msg}")
    if bad:
        print("\nrefusing to write: the source has drifted from every anchor above")
        return 1
    if args.check:
        print(f"\n{len(done)} already applied, {len(todo)} pending")
        return 0

    for path, label, old, new in todo:
        path.write_text(path.read_text().replace(old, new, 1))
        print(f"  [wrote]   {label}")
    print(f"\n{len(todo)} hunks ported, {len(done)} were already in place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
