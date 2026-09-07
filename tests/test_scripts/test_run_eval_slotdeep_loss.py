import json
from pathlib import Path
import subprocess


def test_plan_pins_checkpoints_policies_and_ordinary_python():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["python", "scripts/run_eval_slotdeep_loss.py", "--plan"],
                            cwd=root, check=True, capture_output=True, text=True)
    jobs = json.loads(result.stdout)
    assert [j["gpu"] for j in jobs] == ["0", "1", "2"]
    assert [j["arm"] for j in jobs] == ["t0", "t1", "t2"]
    for job in jobs:
        assert "step1000/training_state.pt" in job["checkpoint"]
        exporter, decoder, diagnostic = job["commands"]
        assert all(c[0] == "python" for c in job["commands"])
        assert "--verify-frozen" in exporter and "--verify-roundtrip" in exporter
        assert "--allow-policy-mismatch" not in decoder
        assert decoder[decoder.index("--gate-rho") + 1] == "3"
        assert decoder[decoder.index("--max-samples") + 1] == "20"
        assert decoder[decoder.index("--max-new-tokens") + 1] == "256"
        assert "--preserve-selector-fp32" in diagnostic
        assert "--include-base-decisions" in diagnostic
        decisions = Path(diagnostic[diagnostic.index("--dump-decisions") + 1])
        metadata = Path(diagnostic[diagnostic.index("--json-output") + 1])
        assert decisions.with_suffix(".json") == metadata


def test_new_holdout_plan_uses_validation_and_all_fixed_policies():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["python", "scripts/run_eval_slotdeep_holdout.py", "--plan"],
                            cwd=root, check=True, capture_output=True, text=True)
    jobs = json.loads(result.stdout)
    assert [j["gpu"] for j in jobs] == ["3", "4", "5", "6", "7"]
    assert [j["label"] for j in jobs] == ["CLOZE7115", "SLOTDEEP7115", "LOSS_T0", "LOSS_T1", "LOSS_T2"]
    for job in jobs:
        c = job["command"]
        assert c[0] == "python"
        assert c[c.index("--prompts") + 1].endswith("/validation.jsonl")
        assert "--allow-policy-mismatch" not in c
        assert c[c.index("--max-new-tokens") + 1] == "256"


def test_single_followup_arm_does_not_rerun_original_suite():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["python", "scripts/run_eval_slotdeep_loss.py", "--plan", "--arms", "t3", "--gpus", "3"],
                            cwd=root, check=True, capture_output=True, text=True)
    jobs = json.loads(result.stdout)
    assert len(jobs) == 1 and jobs[0]["arm"] == "t3" and jobs[0]["gpu"] == "3"
    assert "loss-t3-20260905-step1000" in jobs[0]["checkpoint"]
