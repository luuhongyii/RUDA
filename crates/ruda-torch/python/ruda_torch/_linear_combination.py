import torch

from . import _C


def compute_linear_combination(input, coefficients, *, out=None):
    dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64,
              torch.complex64, torch.complex128, torch.bool, torch.uint8,
              torch.int8, torch.int16, torch.int32, torch.int64)
    if input.device.type != torch._C._get_privateuse1_backend_name() or input.dtype not in dtypes:
        raise RuntimeError('_compute_linear_combination requires a supported RUDA tensor')
    if input.layout != torch.strided or coefficients.layout != torch.strided:
        raise RuntimeError('_compute_linear_combination requires strided tensors')
    dtype = {torch.complex64: torch.float32, torch.complex128: torch.float64}.get(input.dtype, input.dtype)
    if coefficients.device != input.device or coefficients.dtype != dtype:
        raise RuntimeError('coefficients must share the input device and use its real dtype')
    if input.ndim == 0 or coefficients.ndim != 2:
        raise RuntimeError('input must have at least one dimension and coefficients must be a matrix')
    if coefficients.shape[1] > input.shape[0]:
        raise RuntimeError('coefficient reduction dimension exceeds input.shape[0]')
    if out is None:
        if input.numel() == 0:
            raise RuntimeError('Empty tensor not supported')
        out = torch.zeros((coefficients.shape[0],) + input.shape[1:], device=input.device, dtype=input.dtype)
    elif out.device != input.device or out.dtype != input.dtype or out.layout != torch.strided or out.ndim == 0:
        raise RuntimeError('linear-combination output must have matching dtype/device and a non-scalar strided layout')
    _C.check_index_output(out, input, coefficients)
    source = input.resolve_conj().resolve_neg()
    weights = coefficients.resolve_neg()
    target = out.resolve_conj().resolve_neg()
    input_stride, coefficient_stride, count = source.stride(0), weights.stride(1), weights.shape[1]
    source = source.unsqueeze(0)
    sizes, strides = list(source.shape), list(source.stride())
    sizes[1], strides[1] = 1, 0
    source = source.as_strided(sizes, strides)
    weights = weights.as_strided((weights.shape[0],) + (1,) * input.ndim,
                                 (weights.stride(0),) + (0,) * input.ndim)
    destination = target.unsqueeze(1)
    strides = list(destination.stride())
    strides[1] = 0
    destination = destination.as_strided(destination.shape, strides)
    shape = torch.broadcast_shapes(source.shape, weights.shape, destination.shape)
    if tuple(shape) != tuple(destination.shape):
        raise RuntimeError('_compute_linear_combination.out cannot resize or broadcast its output')
    source, weights = source.expand(shape), weights.expand(shape)
    _C.scientific_call(495, [source, weights, destination], [input_stride, coefficient_stride, count], [])
    if target is not out:
        out.copy_(target)
    return out


def register(library):
    library.impl('_compute_linear_combination', compute_linear_combination)
    library.impl('_compute_linear_combination.out', compute_linear_combination)
