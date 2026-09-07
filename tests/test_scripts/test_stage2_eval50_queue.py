import json
from types import SimpleNamespace

import pytest

from scripts import run_stage2_eval50_regen8b as queue


def test_dependency_missing_is_not_training_completion(monkeypatch):
    monkeypatch.setattr(queue.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    with pytest.raises(RuntimeError, match="missing"):
        queue.pane_state()


def test_live_dependency_is_wait_only(monkeypatch):
    monkeypatch.setattr(queue.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="0||123\n"))
    assert queue.pane_state() == dict(dead=False, exit_code=None, pid=123)


def test_gpu_free_requires_three_observations(monkeypatch):
    values = iter([{0: 40000}, {0: 0}, {0: 0}, {0: 0}])
    calls = []
    monkeypatch.setattr(queue, "gpu_memory", lambda: next(values))
    monkeypatch.setattr(queue.time, "sleep", lambda t: calls.append(t))
    queue.wait_free([0])
    assert calls == [15, 15, 15]


def test_comparison_rejects_49_and_wrong_prompts(tmp_path, monkeypatch):
    # Synthetic fixtures only, never placed in real experiment directories.
    monkeypatch.setattr(queue, "WORK", tmp_path)
    monkeypatch.setattr(queue, "PROMPTS", tmp_path / "prompts.json")
    monkeypatch.setattr(queue, "SLOT_RESULT", tmp_path / "slot.json")
    monkeypatch.setattr(queue, "DOMINO_RESULT", tmp_path / "domino.json")
    prompts = [dict(dataset=ds, prompt_index=i, prompt_sha256=f"{ds}/{i}")
        for ds in queue.DATASETS for i in range(50)]
    queue.PROMPTS.write_text(json.dumps(dict(prompts=prompts)))
    def result(modes, proposals):
        return dict(prompt_manifest_sha256=queue.sha(queue.PROMPTS), proposal_slots=proposals,
            arguments=dict(max_new_tokens=256, stop_policy="official"), stop_policy="official",
            results=[dict(r, mode=mode, acceptance_lengths=[2, 3], accepted_sum=5, num_blocks=2)
                     for mode in modes for r in prompts])
    slot = result(["oneshot", "latgate", "oracle16"], 15)
    domino = result(["domino_official"], 16)
    queue.SLOT_RESULT.write_text(json.dumps(slot))
    queue.DOMINO_RESULT.write_text(json.dumps(domino))
    queue.summarize()
    assert json.loads((tmp_path / "comparison.json").read_text())["metrics"]["latgate"]["macro"] == 2.5
    domino["results"].pop()
    queue.DOMINO_RESULT.write_text(json.dumps(domino))
    with pytest.raises(ValueError, match="missing/duplicate/mismatched"):
        queue.summarize()
