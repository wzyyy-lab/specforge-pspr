"""Data-contract tests; synthetic tokens below are not experiment evidence."""
from copy import deepcopy

import pytest

from scripts.prepare_pspr_regen_turns import prepare_row, tokenize_turn


class Tokenizer:
    all_special_ids = [999999]

    def __init__(self):
        self.histories = []

    def apply_chat_template(self, history, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        self.histories.append(deepcopy(history))
        return "".join(m["content"] for m in history) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def encode(self, text, **kwargs):
        assert kwargs == dict(add_special_tokens=False)
        return [ord(c) for c in text]


def row():
    return dict(id="pb-example", status="success", conversations=[
        dict(role="system",content="rules"), dict(role="user",content="Q1"),
        dict(role="assistant",content="answer one"), dict(role="user",content="Q2"),
        dict(role="assistant",content="answer two")])


def test_preserves_history_and_only_supervises_current_answer():
    tok, original = Tokenizer(), row()
    before = deepcopy(original)
    items, counts = prepare_row(original, tok, max_length=3072)
    assert original == before
    assert tok.histories == [before["conversations"][:2], before["conversations"][:4]]
    assert len(items) == counts["kept_turns"] == 2
    for item, text in zip(items, ["answer one", "answer two"]):
        assert item["input_ids"][-len(text):] == list(map(ord,text))
        assert item["loss_mask"] == [0]*item["prefix_tokens"] + [1]*len(text)
        assert item["supervised_tokens"] == len(text)
    assert items[0]["id"] != items[1]["id"]


def test_right_truncation_does_not_change_or_left_truncate_history():
    tok = Tokenizer()
    item, _ = tokenize_turn(tok,[dict(role="user",content="Q")],"ABCDE",max_length=3072)
    truncated, _ = tokenize_turn(tok,[dict(role="user",content="Q")],"ABCDE",
                                 max_length=item["prefix_tokens"]+3)
    assert truncated["input_ids"] == item["input_ids"][:-2]
    assert truncated["supervised_tokens"] == 3 and truncated["truncated"]
    missing, reason = tokenize_turn(tok,[dict(role="user",content="Q")],"ABCDE",max_length=2)
    assert missing is None and reason == "insufficient_current_answer_after_right_truncation"


def test_refuses_thinking_and_broken_success_records():
    example = row()
    example["conversations"][-1]["content"] = "<think>thought</think> answer"
    with pytest.raises(ValueError, match="non-thinking"):
        prepare_row(example, Tokenizer(), max_length=3072)
    example = row()
    example["conversations"][2]["role"] = "user"
    with pytest.raises(ValueError, match="nonalternating"):
        prepare_row(example, Tokenizer(), max_length=3072)


def test_skips_explicit_failed_regeneration():
    items, counts = prepare_row(dict(status="error"), Tokenizer(), max_length=3072)
    assert items == [] and counts["skipped_non_success"] == 1


def test_local_qwen_tokenizer_matches_regen_generation_query():
    from pathlib import Path
    target = Path(__file__).resolve().parents[3] / "TAPS-SP/models/Qwen3-4B"
    if not target.is_dir():
        pytest.skip("local Qwen3 tokenizer unavailable")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(target)
    original = row()
    items, _ = prepare_row(original, tok, max_length=3072)
    for item in items:
        index = item["source_message_index"]
        history = original["conversations"][:index]
        expected = tok.apply_chat_template(history, tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=False)
        assert item["input_ids"][:item["prefix_tokens"]] == expected
        answer = original["conversations"][index]["content"]
        assert tok.decode(item["input_ids"][item["prefix_tokens"]:]) == answer
        assert sum(item["loss_mask"][:item["prefix_tokens"]]) == 0
