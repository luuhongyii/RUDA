import ctypes
from threading import RLock
import torch

_lock = RLock()
_ready = False


def _native():
    global _ready
    from . import _C, _native as library
    with _lock:
        if not _ready:
            if not hasattr(library, 'ruda_torch_varlen_attention_api_version') or not hasattr(_C, 'initialize_varlen_attention'):
                raise RuntimeError('fused varlen attention requires matching rebuilt native libraries')
            library.ruda_torch_varlen_attention_api_version.restype = ctypes.c_uint32
            if library.ruda_torch_varlen_attention_api_version() != 1 or _C.varlen_attention_api_version != 1:
                raise RuntimeError('RUDA varlen attention API mismatch')
            _C.initialize_varlen_attention(ctypes.cast(library.ruda_torch_varlen_attention, ctypes.c_void_p).value)
            _ready = True
    return _C


def _schedule(q, k, cuq, cuk):
    from .varlen_attention import _boundaries
    queries = _boundaries(cuq, q.shape[0])
    keys = _boundaries(cuk, k.shape[0])
    if len(queries) != len(keys):
        raise ValueError('query/key document counts differ')
    records, blocks = [], 0
    for qa, qb, ka, kb in zip(queries, queries[1:], keys, keys[1:]):
        records.append((qa, qb, ka, kb, blocks & 0xffffffff, blocks >> 32))
        blocks += (qb - qa) * q.shape[1] * ((kb - ka + 3) // 4)
        if blocks >= 1 << 64:
            raise OverflowError('varlen attention RNG range exhausted')
    return records, blocks


def reserve(blocks, generator=None):
    from .random import Generator, default_generator
    source = default_generator if generator is None else generator
    if not isinstance(source, Generator):
        raise TypeError('RUDA attention dropout requires a RUDA Generator')
    with source.lock:
        if source.counter + blocks >= 1 << 64:
            raise OverflowError('RUDA attention RNG counter exhausted')
        seed, counter = source.seed_value, source.counter
        source.counter += blocks
    return [seed & 0xffffffff, (seed >> 32) & 0xffffffff, counter & 0xffffffff, counter >> 32]


def _call(op, tensors, policy, rng):
    seed_low, seed_high, base_low, base_high = rng.tolist()
    causal, lower_right, left, right, scale, p, softcap = policy
    _native().varlen_attention(op, tensors, causal, lower_right, left, right, scale, p, softcap,
        seed_low | (seed_high << 32), base_low | (base_high << 32))


@torch.library.custom_op('ruda::varlen_attention_forward', mutates_args=(),
                         tags=(torch.Tag.nondeterministic_seeded,))
def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cuq: torch.Tensor, cuk: torch.Tensor,
            causal: bool, lower_right: bool, left: int, right: int, scale: float, p: float,
            softcap: float, rng: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    records, blocks = _schedule(q, k, cuq, cuk)
    schedule = torch.tensor(records, dtype=torch.int64, device='cpu').to(q.device)
    words = reserve(blocks) if 0 < p < 1 and rng[0].item() < 0 else rng.tolist()[1:]
    state = torch.tensor(words, dtype=torch.int64, device='cpu')
    output = q.new_empty((q.shape[0], q.shape[1], v.shape[2]))
    statistics = torch.empty((q.shape[0] * q.shape[1], 2), device=q.device, dtype=torch.float32)
    _call(0, [q, k, v, schedule, output, statistics], (causal, lower_right, left, right, scale, p, softcap), state)
    return output, statistics, schedule, state


@forward.register_fake
def _forward_fake(q, k, v, cuq, cuk, causal, lower_right, left, right, scale, p, softcap, rng):
    return (q.new_empty((q.shape[0], q.shape[1], v.shape[2])),
        torch.empty((q.shape[0] * q.shape[1], 2), device=q.device, dtype=torch.float32),
        torch.empty((cuq.numel() - 1, 6), device=q.device, dtype=torch.int64),
        torch.empty(4, device='cpu', dtype=torch.int64))


@torch.library.custom_op('ruda::varlen_attention_backward', mutates_args=())
def backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, schedule: torch.Tensor,
             grad: torch.Tensor, stats: torch.Tensor, rng: torch.Tensor,
             causal: bool, lower_right: bool, left: int, right: int, scale: float,
             p: float, softcap: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _call(1, [q, k, v, schedule, grad.contiguous(), stats, dq, dk, dv],
        (causal, lower_right, left, right, scale, p, softcap), rng)
    return dq, dk, dv


@backward.register_fake
def _backward_fake(q, k, v, schedule, grad, stats, rng, causal, lower_right, left, right, scale, p, softcap):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _context(ctx, inputs, output):
    q, k, v, cuq, cuk, *options, requested_rng = inputs
    result, stats, schedule, rng = output
    ctx.save_for_backward(q, k, v, schedule, stats, rng)
    ctx.policy = tuple(options)
    ctx.mark_non_differentiable(stats, schedule, rng)


def _gradient(ctx, grad, _stats, _schedule, _rng):
    q, k, v, schedule, stats, rng = ctx.saved_tensors
    dq, dk, dv = backward(q, k, v, schedule, grad, stats, rng, *ctx.policy)
    return dq, dk, dv, *(None for _ in range(10))


forward.register_autograd(_gradient, setup_context=_context)
