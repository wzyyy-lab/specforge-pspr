"""Gate: the selector CE weight must survive the DDP wrapper.

``DFlashTrainStrategy`` is handed the *wrapped* trainable module. FSDP defines ``__getattr__`` to
forward unknown names to the inner module; ``DistributedDataParallel`` does not. Reading
``selector_loss_alpha`` off the wrapper with a plain ``getattr(..., 0.0)`` therefore returns 0.0
under ``fsdp_sharding: NO_SHARD`` (which ``training/backend.py`` maps to DDP), which drops the
selector cross-entropy from the loss with no error and no warning: joint training silently
degenerates to backbone-only while still logging a healthy ``train/selector_loss``.

Checks:
  A1  DDP-wrapped   -> the configured alpha is seen (the bug this gate exists for)
  A2  unwrapped     -> unchanged (regression guard)
  A3  warmup/ramp   -> the schedule is still honoured through the wrapper
  A4  anti-dead-switch: a raw ``getattr`` on the DDP wrapper really does return the default,
      so A1 is a live test and not a tautology
  A5  the three attribute names the strategy reads are actually the ones the real
      ``OnlineDFlashModel`` defines, so a rename upstream fails here instead of silently

Run: PYTHONPATH=. python scripts/gate_dflash_selector_alpha_wrapper.py
"""

from __future__ import annotations

import inspect
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

ATTRS = ("selector_loss_alpha", "selector_warmup_ratio", "selector_ramp_ratio")


class _FakeDFlashModel(nn.Module):
    def __init__(self, alpha: float, warmup: float, ramp: float) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.selector_loss_alpha = float(alpha)
        self.selector_warmup_ratio = float(warmup)
        self.selector_ramp_ratio = float(ramp)


def main() -> int:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29731")
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP

        from specforge.training.strategies.base import (
            DFlashTrainStrategy,
            StepContext,
        )

        failures = []

        def check(name: str, ok: bool, detail: str) -> None:
            print(f"{name} {'PASS' if ok else 'FAIL'}  {detail}")
            if not ok:
                failures.append(name)

        ctx = StepContext(global_step=10, total_steps=7117)

        # A1 -- the real configuration: alpha 1.0, no warmup, no ramp, DDP-wrapped.
        wrapped = DDP(_FakeDFlashModel(1.0, 0.0, 0.0))
        got = DFlashTrainStrategy(wrapped)._selector_loss_alpha(ctx)
        check(
            "A1",
            got == 1.0,
            f"DDP-wrapped alpha resolves to {got} (must be 1.0; the bug gave 0.0)",
        )

        # A2 -- unwrapped path must be untouched.
        got = DFlashTrainStrategy(_FakeDFlashModel(1.0, 0.0, 0.0))._selector_loss_alpha(
            ctx
        )
        check("A2", got == 1.0, f"unwrapped alpha resolves to {got} (must be 1.0)")

        # A3 -- the schedule must still be read through the wrapper, not defaulted away.
        # warmup_ratio 0.5 of 7117 = 3558 steps, so step 10 is inside warmup.
        wrapped = DDP(_FakeDFlashModel(1.0, 0.5, 0.0))
        early = DFlashTrainStrategy(wrapped)._selector_loss_alpha(ctx)
        late = DFlashTrainStrategy(wrapped)._selector_loss_alpha(
            StepContext(global_step=5000, total_steps=7117)
        )
        check(
            "A3",
            early == 0.0 and late == 1.0,
            f"warmup honoured through wrapper: step10={early}, step5000={late} "
            "(must be 0.0 then 1.0)",
        )

        # A4 -- prove the wrapper really hides the attribute, so A1 is not vacuous.
        raw = getattr(DDP(_FakeDFlashModel(1.0, 0.0, 0.0)), "selector_loss_alpha", 0.0)
        check(
            "A4",
            raw == 0.0,
            f"raw getattr on the DDP wrapper returns {raw} (must be the 0.0 default, "
            "otherwise this gate has no teeth)",
        )

        # A5 -- the names the strategy reads must be the model's real knobs.
        from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel

        params = set(inspect.signature(OnlineDFlashModel.__init__).parameters)
        src = inspect.getsource(DFlashTrainStrategy._selector_loss_alpha)
        missing_model = [a for a in ATTRS if a not in params]
        missing_read = [a for a in ATTRS if a not in src]
        check(
            "A5",
            not missing_model and not missing_read,
            f"attribute names agree (model missing {missing_model}, "
            f"strategy missing {missing_read})",
        )

        print()
        if failures:
            print(f"RESULT: FAIL ({', '.join(failures)})")
            return 1
        print("RESULT: PASS (all checks)")
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
