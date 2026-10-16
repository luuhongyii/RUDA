from functools import partial

import torch

from . import _C


_names = ('erfc', 'lgamma', 'trunc', 'floor', 'ceil', 'acos', 'asin', 'atan',
          'cosh', 'tan', 'sin', 'sinh', 'exp', 'expm1', 'tanh', 'log', 'log10',
          'log2', 'log1p', 'cos', 'sqrt', 'erf', 'sigmoid', 'round', 'frac',
          'reciprocal', 'sign', 'rsqrt', 'neg', 'abs')
_floating = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_integral = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_promoted_integral = _integral + (torch.uint16, torch.uint32, torch.uint64)
_complex = (torch.complex64, torch.complex128)
_real_only = (0, 1, 2, 3, 4, 21, 23, 24, 26)


def _slow(tensors, op, inplace):
    result = _C.foreach_unary_slow(tensors, op, inplace)
    return None if inplace else result


def unary(tensors, *, op, inplace=False):
    if not _C.foreach_unary_fast(tensors):
        return _slow(tensors, op, inplace)
    if any(value.is_conj() or value.is_neg() for value in tensors):
        return _slow(tensors, op, inplace)
    dtype = tensors[0].dtype
    if op == 29 and tensors[0].is_complex():
        return _slow(tensors, op, inplace)
    if op < 28 and dtype in _promoted_integral:
        return _slow(tensors, op, inplace)
    if op == 28 and dtype == torch.bool:
        raise RuntimeError('Negation on a bool tensor is not supported')
    if dtype not in _floating + _integral + _complex or dtype in _complex and op in _real_only:
        raise RuntimeError('unsupported dtype for fused foreach unary operation')
    outputs = tensors if inplace else [torch.empty_like(value) for value in tensors]
    inputs = tensors
    output_views = outputs
    opcode = 449
    if dtype in _complex:
        inputs = [torch.view_as_real(value) for value in tensors]
        output_views = [torch.view_as_real(value) for value in outputs]
        opcode = 450
    flat_inputs = [value.as_strided((value.numel(),), (1,)) for value in inputs]
    flat_outputs = [value.as_strided((value.numel(),), (1,)) for value in output_views]
    _C.scientific_call(opcode, [*flat_inputs, *flat_outputs], [op, len(tensors)], [])
    if inplace:
        if op != 30:
            _C.foreach_unary_bump(tensors)
        return None
    return outputs


def register(library):
    for op, name in enumerate(_names):
        library.impl('_foreach_' + name, partial(unary, op=op))
        library.impl('_foreach_' + name + '_', partial(unary, op=op, inplace=True))
    library.impl('_foreach_zero_', partial(unary, op=30, inplace=True))
