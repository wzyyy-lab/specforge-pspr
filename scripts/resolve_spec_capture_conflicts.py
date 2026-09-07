#!/usr/bin/env python3
"""Resolve the 3-way merge conflicts left by the spec-capture patch on sglang 2dd449ce5.

How we got here
---------------
`managed_local` training refuses to start without `sglang.srt.spec_capture_sink`
(`specforge/launch_plan.py:962-971`).  The installed sglang is an upstream checkout at 2dd449ce5,
which matches neither shipped patch target, and the previously patched tree is gone -- the same
environment change that made the historical decode baselines irreproducible.

Three approaches were tried and rejected:
  * `scripts/apply_sglang_spec_capture_patch.sh` runs `git apply -p2` with cwd `<root>/python`, and
    because that directory is INSIDE the sglang git repo, git resolves the patch paths against the
    repo root, finds them outside cwd, prints "Skipped patch ..." for every file and exits 0 -- so
    the script reports "applied" for a patch that never landed.
  * `patch -p1 --fuzz=3` places 31/38 hunks but silently misplaces several: it put ServerArgs fields
    inside a method body (orphaning an `assert`), `drain_spec_captures()` inside the signature of
    `add_external_corpus`, and a capture block inside `_apply_decode_logprobs`'s signature.  Three
    of those were IndentationError/SyntaxError; a fourth kind would have been silent.
  * `git apply -p1 --reject` (exact context) rejects ~20 hunks, which is a hand-port of the whole
    patch.

What works: `git fetch --depth=1 origin refs/tags/v0.5.18` brings the pre-image blobs the patch's
`index` lines name, after which `git apply -p1 -3` performs a real 3-way merge -- every hunk lands
in the right place and the 12 genuine conflicts carry ours/theirs markers.  This script resolves
those 12.

Each resolution keeps the LOCAL structure and adds only the capture behaviour, so nothing upstream
changes semantics: dropped from "theirs" are `ReturnHiddenStatesMode` (absent here),
`flush_trace_batch`, a second `maybe_send_health_check_signal()` (this version already calls it in
`process_batch_result`), `return_sampling_mask`, and `free_group_begin/end` in the prefill path
(this version does not batch frees there).

Run: sudo -n python3 scripts/resolve_spec_capture_conflicts.py [--check]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path("/sgl-workspace/sglang/python/sglang/srt")

# (file, marker text that must appear inside the conflict, replacement for the whole conflict block)
RESOLUTIONS: list[tuple[str, str, str]] = [
    (
        "layers/logits_processor.py",
        "last_hidden_states=last_hidden_states_to_store,",
        """                # FIXME: These fields are not logits-related but are passed through here as a
                # workaround since ForwardBatch is local to forward_batch_generation().
                # They should be moved to GenerationBatchResult to keep this class clean.
                last_hidden_states=last_hidden_states_to_store,
""",
    ),
    (
        "managers/io_struct.py",
        "spec_capture: Optional[Union[List[Optional[Dict]], Dict]] = None",
        """    return_hidden_states: Union[List[bool], bool] = False
    # Spec-training capture sink instructions (see spec_capture_sink.py).
    # Batch-level: List[Optional[dict]]; per-request after __getitem__.
    spec_capture: Optional[Union[List[Optional[Dict]], Dict]] = None
""",
    ),
    (
        "managers/schedule_batch.py",
        "self.spec_capture_result = None",
        """        self.spec_capture_aux: List[torch.Tensor] = []
        self.spec_capture_last_hidden: List[torch.Tensor] = []
        self.spec_capture_result = None  # per-request sink result -> output field
""",
    ),
    (
        "managers/scheduler.py",
        "self.future_map.stash(future_indices, stash_payload)",
        """                            stash_payload = (
                                batch_result.next_draft_input
                                if not batch.spec_algorithm.is_none()
                                else batch_result.next_token_ids
                            )
                            self.future_map.stash(future_indices, stash_payload)
                            batch_result.copy_to_cpu(
                                return_logprob=batch.return_logprob,
                                return_hidden_states=self._should_copy_hidden_states_to_cpu(
                                    batch
                                ),
                            )
""",
    ),
    (
        "managers/scheduler.py",
        "return_logprob=cur_batch.return_logprob,",
        """                return_logprob=self.cur_batch.return_logprob,
                return_hidden_states=self._should_copy_hidden_states_to_cpu(
                    self.cur_batch
                ),
""",
    ),
    (
        "managers/scheduler.py",
        "flush_trace_batch(batch.reqs)",
        """        # Complete capture transfers on the scheduler thread before processing
        # any kind of next result (prefill, decode, or idle).  This preserves
        # the output socket's single-thread ownership under mixed traffic.
        self.batch_result_processor.drain_spec_captures()

""",
    ),
    (
        "managers/scheduler.py",
        "# Flush any health-check signal deferred while the engine was busy.",
        """        # A capture response is intentionally delayed until its background
        # sink transfer completes; finish it on this thread so ownership of
        # the ZeroMQ output socket remains single-threaded.  Do not enter the
        # idle sleeper while a future is outstanding or the producer waiting
        # for that response could deadlock.
        self.batch_result_processor.drain_spec_captures()
        if self.batch_result_processor.has_pending_spec_captures():
            time.sleep(0.001)
            return
""",
    ),
    (
        "managers/scheduler_components/batch_result_processor.py",
        "pending_spec_captures: List = []",
        """        pending_spec_captures: List = []
""",
    ),
    (
        "managers/scheduler_components/batch_result_processor.py",
        "capture_hidden_mode=prefill_hidden_capture_mode,",
        """                if req.finished() or req.is_retracted:
                    # decode req in mixed batch or retracted req
""",
    ),
    (
        "managers/scheduler_components/batch_result_processor.py",
        "if req.return_sampling_mask:",
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
                            )

                    if req.spec_capture is not None and req.finished():
                        pending = self._sink_spec_capture(req)
                        if pending is not None:
                            pending_spec_captures.append(pending)
""",
    ),
    (
        "managers/scheduler_components/batch_result_processor.py",
        "capture_req_ids = {id(item[0]) for item in pending_spec_captures}",
        """        if pending_spec_captures:
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
""",
    ),
    (
        "managers/scheduler_components/output_streamer.py",
        'req.hidden_states[-1] if req.hidden_states else None',
        """            self.output_hidden_states.append(
                req.hidden_states if req.return_hidden_states else None
            )
        # Per-request spec-capture result (aligned with rids), like the field above.
        self.spec_capture.append(getattr(req, "spec_capture_result", None))
""",
    ),
    (
        "server_args.py",
        'NS("exec.features"),',
        # Keep the whole local block; the three capture options are declared in the local plain
        # dataclass style by scripts/port_spec_capture_hunks.py, which runs after this script.
        "OURS",
    ),
]

CONFLICT = re.compile(
    r"^<<<<<<< ours\n(?P<ours>.*?)^=======\n(?P<theirs>.*?)^>>>>>>> theirs\n",
    re.MULTILINE | re.DOTALL,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    by_file: dict[str, list[tuple[str, str]]] = {}
    for rel, marker, repl in RESOLUTIONS:
        by_file.setdefault(rel, []).append((marker, repl))

    total, failed = 0, []
    for rel, rules in by_file.items():
        path = ROOT / rel
        text = path.read_text()
        blocks = list(CONFLICT.finditer(text))
        if not blocks:
            print(f"  [clean]   {rel} (no conflicts left)")
            continue
        used = set()
        out, last = [], 0
        for m in blocks:
            body = m.group(0)
            hit = None
            for i, (marker, repl) in enumerate(rules):
                if marker in body and i not in used:
                    hit = (i, repl)
                    break
            if hit is None:
                failed.append(f"{rel}: no resolution matches a conflict at offset {m.start()}")
                continue
            i, repl = hit
            used.add(i)
            out.append(text[last:m.start()])
            out.append(m.group("ours") if repl == "OURS" else repl)
            last = m.end()
            total += 1
        out.append(text[last:])
        merged = "".join(out)
        print(f"  [{'would resolve' if args.check else 'resolved'}] {rel}: "
              f"{len(used)}/{len(blocks)} conflicts")
        if not args.check:
            path.write_text(merged)

    if failed:
        for msg in failed:
            print(f"  [FAIL]    {msg}")
        return 1
    print(f"\n{total} conflicts {'would be ' if args.check else ''}resolved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
