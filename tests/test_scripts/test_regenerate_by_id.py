import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import regenerate_by_id as driver
from scripts import regenerate_train_data as regen


def test_resume_uses_ids_not_completed_line_count(tmp_path):
    rows = [dict(id="second", status="success"), dict(id="fourth", status="skipped")]
    (tmp_path / "journal-1.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows) + '{"id":"first"')
    progress = driver.read_journals(tmp_path, {"first", "second", "third", "fourth"})
    assert set(progress) == {"second", "fourth"}
    assert [key for key in ("first", "second", "third", "fourth") if key not in progress] == ["first", "third"]
    assert (tmp_path / "journal-1.jsonl").read_text().endswith('{"id":"first"')


def test_resume_rejects_corruption_and_wrong_ids(tmp_path):
    path = tmp_path / "journal-1.jsonl"
    path.write_text('{"id":"alien","status":"success"}\n')
    with pytest.raises(ValueError, match="unexpected"):
        driver.read_journals(tmp_path, {"expected"})
    path.write_text("{BROKEN}\n")
    with pytest.raises(json.JSONDecodeError):
        driver.read_journals(tmp_path, {"expected"})


def test_source_duplicate_id_rejected(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_text('{"id":"x"}\n{"id":"x"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        list(driver.input_rows(path))


def test_regenerated_multi_turn_history_and_metadata(monkeypatch):
    queries = []
    def create(**kwargs):
        queries.append(copy.deepcopy(kwargs))
        answer = "2" if len(queries) == 1 else "5"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer),
            finish_reason="stop")], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1))
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(regen, "OpenAI", lambda **kwargs: fake)
    args = SimpleNamespace(model="Qwen3-8B", temperature=0.0, reasoning="disable", max_tokens=2048,
        top_p=None, top_k=None, repetition_penalty=None, is_gpt_oss=False,
        record_generation_metadata=True, api_timeout=300.0)
    row = dict(id="test", conversations=[dict(role="user", content="1+1?"),
        dict(role="assistant", content="ORIGINAL BAD ANSWER"), dict(role="user", content="Add 3.")])
    result = regen.call_sglang(args, "127.0.0.1:32100", row)
    assert result["status"] == "success"
    assert queries[1]["messages"][1]["content"] == "2"
    assert "ORIGINAL BAD ANSWER" not in json.dumps(queries)
    assert all("generation" not in message for query in queries for message in query["messages"])
    assert queries[0]["temperature"] == 0.0
    assert queries[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert result["conversations"][1]["generation"]["finish_reason"] == "stop"


def test_legacy_call_has_no_generation_metadata(monkeypatch):
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Hello"))]))))
    monkeypatch.setattr(regen, "OpenAI", lambda **kwargs: fake)
    args = SimpleNamespace(model="old", temperature=0.0, reasoning="disable", max_tokens=8,
        top_p=None, top_k=None, repetition_penalty=None, is_gpt_oss=False)
    result = regen.call_sglang(args, "unused", dict(conversations=[dict(role="user", content="Hi")]))
    assert result["conversations"][-1] == dict(role="assistant", content="Hello")


def test_materialize_preserves_source_order_and_exclusion_accounting(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "regen.jsonl"
    inputs = [dict(id=str(i), conversations=[dict(role="user", content=f"q{i}"),
        dict(role="assistant", content="old")]) for i in range(3)]
    source.write_text("".join(json.dumps(row)+"\n" for row in inputs))
    def good(row):
        item = copy.deepcopy(row)
        item["status"] = "success"
        item["conversations"][1] = dict(role="assistant", content="new",
            generation=dict(finish_reason="length"))
        return item
    results = {"2": good(inputs[2]), "0": good(inputs[0]),
        "1": dict(inputs[1], status="error", error="retained separately")}
    args = SimpleNamespace(input_file_path=source, output_file_path=output,
        num_samples=None, model="test", max_tokens=2048)
    driver.materialize(args, results)
    assert [json.loads(line)["id"] for line in output.read_text().splitlines()] == ["0", "2"]
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["counts"] == {"success": 2, "error": 1}
    assert manifest["status"] == "complete_with_exclusions"
    assert manifest["assistant_finish_reasons"] == {"length": 2}
    # Resume after publication can reuse exact files, never overwrite different data.
    driver.materialize(args, results)
    output.write_text("unrelated content\n")
    with pytest.raises(FileExistsError):
        driver.materialize(args, results)


def test_full_driver_resumes_out_of_order_ids(tmp_path, monkeypatch):
    source, target = tmp_path / "source.jsonl", tmp_path / "regen.jsonl"
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    original = [dict(id=str(i), conversations=[dict(role="user", content=f"q{i}"),
        dict(role="assistant", content="old")]) for i in range(10)]
    source.write_text("".join(json.dumps(row)+"\n" for row in original))
    def good(row):
        result = copy.deepcopy(row)
        result["status"] = "success"
        result["conversations"][1] = dict(role="assistant", content="new",
            generation=dict(finish_reason="stop"))
        return result
    progress = target.with_suffix(".progress")
    progress.mkdir()
    (progress / "journal-1.jsonl").write_text(json.dumps(good(original[7]))+"\n")
    calls = []
    def fake(args, address, row):
        calls.append(row["id"])
        return good(row)
    monkeypatch.setattr(driver, "regenerate_one", fake)
    monkeypatch.setattr("sys.argv", ["regenerate_by_id.py", "--model", str(model),
        "--server-address", "fake0", "fake1", "--concurrency", "2",
        "--input-file-path", str(source), "--output-file-path", str(target)])
    driver.main()
    assert set(calls) == {str(i) for i in range(10)} - {"7"}
    assert [json.loads(line)["id"] for line in target.read_text().splitlines()] == [str(i) for i in range(10)]
    calls.clear()
    driver.main()
    assert not calls
