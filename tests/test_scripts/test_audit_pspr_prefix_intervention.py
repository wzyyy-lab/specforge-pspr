"""Synthetic pair bookkeeping, not model-quality evidence."""
from scripts.audit_pspr_prefix_intervention import make_pair


def test_pair_changes_prefix_only_and_keeps_scored_tokens():
    base,changed,positions,shifted=make_pair([9,1,2,7,8,0],[0,0,0,1,1,1],3,5,[3,4])
    assert base==[9,1,2,7,8]
    assert changed==[9,1,2,3,4,7,8]
    assert positions==[3,4] and shifted==[5,6]
    assert [base[i] for i in positions]==[changed[i] for i in shifted]
