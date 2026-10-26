import importlib.util
import itertools
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('ruda_broadcast_host', Path(__file__).resolve().parents[1] / 'ruda_torch/_broadcast.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
broadcast_shapes = module.broadcast_shapes


def test_static_broadcast_matches_torch():
    shapes = [(), (0,), (1,), (2,), (3,), (2, 1), (1, 3), (0, 3), (2, 3), (1, 2, 3)]
    for values in itertools.product(shapes, repeat=3):
        try:
            expected = torch.broadcast_shapes(*values)
        except RuntimeError:
            with pytest.raises(RuntimeError):
                broadcast_shapes(*values)
        else:
            assert broadcast_shapes(*values) == expected
    assert broadcast_shapes() == torch.Size()
