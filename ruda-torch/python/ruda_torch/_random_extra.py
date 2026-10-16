from contextlib import contextmanager

import torch

from . import _C
from .random import Generator, default_generator


@contextmanager
def _reservation(generator, blocks):
    generator = default_generator if generator is None else generator
    if isinstance(generator, Generator):
        with generator.lock:
            if generator.counter + blocks >= 1 << 64:
                raise OverflowError('RUDA random counter exhausted')
            counter = generator.counter
            generator.counter += blocks
            seed = generator.seed_value
            yield [seed & 0xffffffff, seed >> 32, counter & 0xffffffff, counter >> 32]
    elif isinstance(generator, torch.Generator):
        yield _C.random_launch_seed(generator)
    else:
        raise TypeError('expected a RUDA Generator or torch.Generator on the RUDA device')


def _storage(a):
    if a.device.type != 'ruda' or a.layout != torch.strided:
        raise RuntimeError('random filling requires strided RUDA storage')


def _launch(a, op, parameters, scalars, blocks):
    _C.check_inplace(a, a)
    target = torch.empty_like(a) if a.is_conj() or a.is_neg() else a
    with _reservation(parameters[0], blocks) as state, torch.no_grad():
        _C.scientific_call(op, [target], [*state, *parameters[1:]], scalars)
        if target is not a:
            a.copy_(target)
    return a


def random_(a, generator=None, *, bounded=False, low=0, high=None):
    _storage(a)
    bounds = _C.random_bounds(a, bounded, low, high)
    if not a.numel():
        return a
    words = a.numel() * (2 if bounds[3] else 1)
    return _launch(a, 414, [generator, *bounds], [], (words + 3) // 4)


def random_from(a, low, high, *, generator=None):
    return random_(a, generator, bounded=True, low=low, high=high)


def random_to(a, high, *, generator=None):
    return random_(a, generator, bounded=True, high=high)


def _distribution(a, op, first, second, generator):
    _storage(a)
    if not a.numel():
        return a
    supported = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if op == 416:
        supported += (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if a.dtype not in supported:
        raise RuntimeError('unsupported dtype for random distribution')
    words = a.numel() * (2 if a.dtype == torch.float64 else 1)
    blocks = (words + 3) // 4
    return _launch(a, op, [generator], [first, second], blocks)


def exponential_(a, lambd=1, *, generator=None):
    lambd = float(lambd)
    if not lambd > 0:
        raise RuntimeError('exponential_ requires lambda > 0')
    return _distribution(a, 415, lambd, 0.0, generator)


def geometric_(a, p, *, generator=None):
    p = float(p)
    if not 0 < p < 1:
        raise RuntimeError('geometric_ requires 0 < p < 1')
    return _distribution(a, 416, p, 0.0, generator)


def cauchy_(a, median=0, sigma=1, *, generator=None):
    median, sigma = float(median), float(sigma)
    if not sigma > 0 or not a.is_floating_point():
        raise RuntimeError('cauchy_ requires floating output and sigma > 0')
    return _distribution(a, 417, median, sigma, generator)


def log_normal_(a, mean=1, std=2, *, generator=None):
    mean, std = float(mean), float(std)
    if not std > 0:
        raise RuntimeError('log_normal_ requires std > 0')
    return _distribution(a, 418, mean, std, generator)


def register(library):
    library.impl('random_', random_)
    library.impl('random_.from', random_from)
    library.impl('random_.to', random_to)
    library.impl('exponential_', exponential_)
    library.impl('geometric_', geometric_)
    library.impl('cauchy_', cauchy_)
    library.impl('log_normal_', log_normal_)
    from ._multinomial import multinomial
    library.impl('multinomial', multinomial)
    library.impl('multinomial.out', multinomial)
    from ._randperm import randperm_out
    library.impl('randperm.generator_out', randperm_out)
    from ._poisson import poisson
    library.impl('poisson', poisson)
    from ._distribution_family import register as register_distributions
    register_distributions(library)
