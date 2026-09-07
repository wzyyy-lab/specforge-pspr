import random

import pytest

from specforge.data.prompt_builder import (
    _normalize_integer_sequence,
    _normalize_integer_sequence_general,
)


@pytest.mark.parametrize("field,binary", [("input_ids",False),("loss_mask",True)])
def test_fast_path_is_identical_to_original_validation(field,binary):
    rng = random.Random(42)
    cases = [[],[1],[0,1,1],[[1,2]],[True,False],[1,2.0],[[1],[2]],(0,1),"123",None]
    cases += [[rng.randrange(2 if binary else 151936) for _ in range(3072)] for _ in range(10)]
    for value in cases:
        kwargs = dict(field=field,source="contract-test",binary=binary)
        try:
            expected = _normalize_integer_sequence_general(value,**kwargs)
        except ValueError as exc:
            with pytest.raises(ValueError) as actual:
                _normalize_integer_sequence(value,**kwargs)
            assert str(actual.value) == str(exc)
        else:
            actual = _normalize_integer_sequence(value,**kwargs)
            assert actual == expected and actual is not value
