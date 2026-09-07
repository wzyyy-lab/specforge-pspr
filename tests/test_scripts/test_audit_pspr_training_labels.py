"""Synthetic prefix bookkeeping only; not experiment evidence."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location("label_audit",Path(__file__).resolve().parents[2]/"scripts/audit_pspr_training_labels.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_prefix_flags_track_observed_serialized_headers():
    ids=[9,1,2,7,7,1,2,3,4,7]
    flags=module.assistant_prefix_flags(ids,[1,2],[3,4])
    assert flags[0] is None
    assert flags[3] is False and flags[4] is False
    assert flags[7] is True and flags[9] is True

