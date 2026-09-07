import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.prepare_pspr_holdout import prepare, prompt_key, user_turns


def test_normalization_and_all_user_turns():
    assert prompt_key(" Ａ  b\nC ") == prompt_key("a B c")
    assert list(user_turns(dict(conversations=[
        dict(role="user", content="first"), dict(role="assistant", content="answer"),
        dict(from_="ignored"), dict(**{"from": "human", "value": "second"}),
    ]))) == ["first", "second"]
    with pytest.raises(ValueError, match="non-text"):
        list(user_turns(dict(messages=[dict(role="user", content=["image"])])))


def test_disjoint_reproducible_excluded_prompt_draw(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    records = []
    for i in range(30):
        records.append(dict(source="a" if i < 15 else "b", conversations=[
            {"from": "human", "value": f"Question {i}"},
            {"from": "gpt", "value": "Answer"},
        ]))
    # A differently cased duplicate must not appear twice across the splits.
    records.append(dict(source="b", conversations=[
        {"from": "human", "value": "QUESTION  29"}, {"from": "gpt", "value": "Answer"}]))
    pq.write_table(pa.Table.from_pylist(records), source / "train-00000.parquet")
    excluded = tmp_path / "excluded.jsonl"
    excluded.write_text(json.dumps(dict(id="pb-0-0", conversations=[
        dict(role="user", content="unrelated"), dict(role="assistant", content="answer"),
        dict(role="user", content="QUESTION 1"),
    ])) + "\n")
    a = prepare(source, [excluded], tmp_path / "a", 10, 42, 8000)
    b = prepare(source, [excluded], tmp_path / "b", 10, 42, 8000)
    assert a["eligible_unique"] == 28
    assert a["selections"] == b["selections"]
    assert "pb-0-0" not in a["selections"] and "pb-0-1" not in a["selections"]
    for split in ("calibration", "validation"):
        assert a["outputs"][split]["sha256"] == b["outputs"][split]["sha256"]
        assert a["outputs"][split]["count"] == 10
    with pytest.raises(FileExistsError):
        prepare(source, [excluded], tmp_path / "a", 10, 42, 8000)
    with pytest.raises(ValueError, match="unique eligible"):
        prepare(source, [excluded], tmp_path / "too_large", 20, 42, 8000)
    assert not (tmp_path / "too_large").exists()
