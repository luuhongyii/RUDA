"""Block-scaled GEMMs and step-boundary delayed FP8 scale histories."""
from __future__ import annotations

import ctypes
import copy
import torch
from torch import nn
from torch.autograd.function import once_differentiable
from .float8_training import _FLOATS, _FP8, _STRATEGIES, _lock, _encode, _mm, _power_of_two_scale

_block_ready = False
_shard_ready = False


def _block_native():
    global _block_ready
    from . import _C, _native
    with _lock:
        if not _block_ready:
            if not hasattr(_native, 'ruda_torch_float8_block_api_version') or not hasattr(_C, 'initialize_float8_block'):
                raise RuntimeError('block FP8 requires matching rebuilt RUDA libraries')
            _native.ruda_torch_float8_block_api_version.restype = ctypes.c_uint32
            if _native.ruda_torch_float8_block_api_version() != 1 or _C.float8_block_api_version != 1:
                raise RuntimeError('RUDA block FP8 API mismatch')
            _C.initialize_float8_block(ctypes.cast(_native.ruda_torch_float8_block, ctypes.c_void_p).value)
            _block_ready = True
    return _C


@torch.library.custom_op('ruda::float8_block_quantize', mutates_args=())
def _block_encode(x: torch.Tensor, dtype: torch.dtype, rows: int, cols: int,
                  margin: int, power_of_two: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if x.device.type != 'ruda' or x.ndim != 2 or x.dtype not in _FLOATS or dtype not in _FP8:
        raise ValueError('block FP8 encoding requires a floating RUDA matrix')
    if rows < 1 or cols < 32 or cols % 32 or not 0 <= margin <= 16:
        raise ValueError('invalid FP8 block geometry or margin')
    encoded = torch.empty(x.shape, device=x.device, dtype=dtype)
    scales = torch.empty(((x.shape[0] + rows - 1) // rows, (x.shape[1] + cols - 1) // cols),
                         dtype=torch.float32, device=x.device)
    _block_native().float8_block_dispatch(1, [x.contiguous(), encoded, scales], rows, rows, cols, margin, power_of_two, 0)
    return encoded, scales


@_block_encode.register_fake
def _block_encode_fake(x, dtype, rows, cols, margin, power_of_two):
    return (torch.empty(x.shape, device=x.device, dtype=dtype),
            torch.empty(((x.shape[0] + rows - 1) // rows, (x.shape[1] + cols - 1) // cols), device=x.device, dtype=torch.float32))


@torch.library.custom_op('ruda::float8_block_mm', mutates_args=())
def _block_mm(a: torch.Tensor, weight: torch.Tensor, scale_a: torch.Tensor, scale_w: torch.Tensor,
              rows_a: int, rows_w: int, block_k: int, strategy: int) -> torch.Tensor:
    if a.ndim != 2 or weight.ndim != 2 or a.shape[1] != weight.shape[1] or a.dtype not in _FP8 or weight.dtype not in _FP8:
        raise ValueError('block FP8 GEMM requires [M,K] and [N,K] encoded matrices')
    if rows_a < 1 or rows_w < 1 or block_k < 32 or block_k % 32 or strategy not in (0, 1, 2):
        raise ValueError('invalid block FP8 GEMM geometry or strategy')
    if a.device.type != 'ruda' or any(t.device != a.device for t in (weight, scale_a, scale_w)):
        raise ValueError('block FP8 operands must share the RUDA device')
    for scale, matrix, rows in ((scale_a, a, rows_a), (scale_w, weight, rows_w)):
        if scale.dtype != torch.float32 or scale.shape != ((matrix.shape[0] + rows - 1) // rows, (matrix.shape[1] + block_k - 1) // block_k):
            raise ValueError('block FP8 scale geometry differs')
    output = torch.empty((a.shape[0], weight.shape[0]), dtype=torch.float32, device=a.device)
    _block_native().float8_block_dispatch(0, [a.contiguous(), weight.contiguous(), scale_a.contiguous(), scale_w.contiguous(), output],
                                        rows_a, rows_w, block_k, 0, False, strategy)
    return output


@_block_mm.register_fake
def _block_mm_fake(a, weight, scale_a, scale_w, rows_a, rows_w, block_k, strategy):
    return torch.empty((a.shape[0], weight.shape[0]), dtype=torch.float32, device=a.device)


def _shard_native():
    global _shard_ready
    from . import _C, _native
    with _lock:
        if not _shard_ready:
            if not hasattr(_native, 'ruda_torch_float8_shard_api_version') or not hasattr(_C, 'initialize_float8_shard'):
                raise RuntimeError('cross-block TP FP8 requires matching rebuilt RUDA libraries')
            _native.ruda_torch_float8_shard_api_version.restype = ctypes.c_uint32
            if _native.ruda_torch_float8_shard_api_version() != 1 or _C.float8_shard_api_version != 1:
                raise RuntimeError('RUDA sharded block FP8 API mismatch')
            _C.initialize_float8_shard(ctypes.cast(_native.ruda_torch_float8_shard, ctypes.c_void_p).value)
            _shard_ready = True
    return _C


def _shard_geometry(x, rows, cols, origin_row, origin_col, global_rows, global_cols):
    if x.ndim != 2 or x.dtype not in _FLOATS or x.device.type != 'ruda':
        raise ValueError('sharded block FP8 requires a floating RUDA matrix')
    if rows < 1 or cols < 32 or cols % 32 or min(origin_row, origin_col, global_rows, global_cols) < 0:
        raise ValueError('invalid sharded FP8 block geometry')
    if origin_row + x.shape[0] > global_rows or origin_col + x.shape[1] > global_cols:
        raise ValueError('FP8 shard lies outside its logical matrix')
    return (2, (global_rows + rows - 1) // rows, (global_cols + cols - 1) // cols)


@torch.library.custom_op('ruda::float8_shard_statistics', mutates_args=())
def _shard_statistics(x: torch.Tensor, rows: int, cols: int, origin_row: int, origin_col: int,
                       global_rows: int, global_cols: int) -> torch.Tensor:
    shape = _shard_geometry(x, rows, cols, origin_row, origin_col, global_rows, global_cols)
    statistics = torch.empty(shape, device=x.device, dtype=torch.float32)
    _shard_native().float8_shard_dispatch(0, [x.contiguous(), statistics],
        [rows, cols, origin_row, origin_col, global_rows, global_cols])
    return statistics


@_shard_statistics.register_fake
def _shard_statistics_fake(x, rows, cols, origin_row, origin_col, global_rows, global_cols):
    shape = _shard_geometry(x, rows, cols, origin_row, origin_col, global_rows, global_cols)
    return torch.empty(shape, device=x.device, dtype=torch.float32)


@torch.library.custom_op('ruda::float8_shard_quantize', mutates_args=())
def _shard_encode(x: torch.Tensor, statistics: torch.Tensor, dtype: torch.dtype,
                  rows: int, cols: int, origin_row: int, origin_col: int,
                  global_rows: int, global_cols: int, margin: int, power_of_two: bool
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    shape = _shard_geometry(x, rows, cols, origin_row, origin_col, global_rows, global_cols)
    if statistics.shape != shape or statistics.dtype != torch.float32 or statistics.device != x.device:
        raise ValueError('FP8 statistics do not match the global block grid')
    if dtype not in _FP8 or not 0 <= margin <= 16:
        raise ValueError('invalid sharded FP8 format or margin')
    height, width = x.shape[0] + origin_row % rows, x.shape[1] + origin_col % cols
    encoded = torch.empty((height, width), device=x.device, dtype=dtype)
    scales = torch.empty(((height + rows - 1) // rows, (width + cols - 1) // cols), device=x.device, dtype=torch.float32)
    _shard_native().float8_shard_dispatch(1, [x.contiguous(), statistics.contiguous(), encoded, scales],
        [rows, cols, origin_row, origin_col, global_rows, global_cols, margin, int(power_of_two)])
    return encoded, scales


@_shard_encode.register_fake
def _shard_encode_fake(x, statistics, dtype, rows, cols, origin_row, origin_col,
                       global_rows, global_cols, margin, power_of_two):
    height, width = x.shape[0] + origin_row % rows, x.shape[1] + origin_col % cols
    return (torch.empty((height, width), device=x.device, dtype=dtype),
            torch.empty(((height + rows - 1) // rows, (width + cols - 1) // cols), device=x.device, dtype=torch.float32))


def _block_operand(value, dtype, rows, recipe, group, axis, replicas):
    value = value.detach().contiguous()
    cols = recipe.block_size
    if group is None or group.world_size == replicas or axis == -1 or value.shape[axis] % (rows if axis == 0 else cols) == 0:
        encoded, scales = _block_encode(value, dtype, rows, cols, recipe.margin, recipe.power_of_two_scale)
        return encoded, scales, 0
    from ._collective_ops import collective_reduce, group_key
    origin = [0, 0]
    full = list(value.shape)
    origin[axis] = value.shape[axis] * (group.rank // replicas)
    full[axis] *= group.world_size // replicas
    geometry = (rows, cols, *origin, *full)
    statistics = _shard_statistics(value, *geometry)
    statistics = collective_reduce(statistics, 1, group_key(group))
    encoded, scales = _shard_encode(value, statistics, dtype, *geometry, recipe.margin, recipe.power_of_two_scale)
    return encoded, scales, origin[0] % rows


def _block_product(a, weight, recipe, role_a, role_w, group=None, axis_a=-1, axis_w=-1, replicas=1):
    formats = (recipe.input_dtype, recipe.weight_dtype, recipe.grad_dtype)
    rows = (recipe.input_block_rows, recipe.weight_block_rows, recipe.grad_block_rows)
    qa, sa, pad_a = _block_operand(a, formats[role_a], rows[role_a], recipe, group, axis_a, replicas)
    qw, sw, pad_w = _block_operand(weight, formats[role_w], rows[role_w], recipe, group, axis_w, replicas)
    output = _block_mm(qa, qw, sa, sw, rows[role_a], rows[role_w], recipe.block_size, _STRATEGIES[recipe.strategy])
    return output.narrow(0, pad_a, a.shape[0]).narrow(1, pad_w, weight.shape[0]).contiguous()


@torch.library.custom_op('ruda::float8_delayed_quantize', mutates_args=())
def _delayed_observe(x: torch.Tensor, scale: torch.Tensor, pending: torch.Tensor,
                     dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale.shape != () or pending.shape != (3,) or scale.dtype != torch.float32 or pending.dtype != torch.float32:
        raise ValueError('delayed FP8 requires scalar scale and [amax,invalid,seen] accumulator')
    if scale.device != x.device or pending.device != x.device:
        raise ValueError('delayed FP8 state must be on the operand device')
    updated = pending.clone()
    if x.numel():
        maximum = x.detach().float().abs().amax()
        finite = torch.isfinite(maximum)
        observed = torch.stack((torch.where(finite, maximum, torch.zeros_like(maximum)), (~finite).float(), torch.ones_like(maximum)))
        updated = torch.maximum(pending, observed)
    used_scale = scale.clone()
    return _encode(x, used_scale, dtype), used_scale, updated


@_delayed_observe.register_fake
def _delayed_encode_fake(x, scale, pending, dtype):
    return torch.empty(x.shape, device=x.device, dtype=dtype), torch.empty_like(scale), torch.empty_like(pending)


def _delayed_encode(x, scale, pending, dtype):
    encoded, used_scale, updated = _delayed_observe(x, scale, pending, dtype)
    pending.copy_(updated)
    return encoded, used_scale


@torch.library.custom_op('ruda::float8_delayed_advance', mutates_args=())
def _delayed_advance(scales: torch.Tensor, history: torch.Tensor, observed: torch.Tensor, step: torch.Tensor,
                     interval: int, history_max: bool, power_of_two: bool,
                     input_maximum: float, weight_maximum: float, grad_maximum: float
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seen = observed[:, 2] > 0
    rolled = torch.cat((observed[:, None, :2], history[:, :-1]), dim=1)
    updated_history = torch.where(seen[:, None, None], rolled, history)
    statistics = updated_history.amax(1) if history_max else updated_history[:, 0]
    maximum = torch.tensor([input_maximum, weight_maximum, grad_maximum], device=scales.device, dtype=torch.float32)
    proposed = torch.where((statistics[:, 0] > 0) & (statistics[:, 1] == 0),
        (statistics[:, 0] / maximum).clamp_min(torch.finfo(torch.float32).tiny), torch.ones_like(scales))
    if power_of_two:
        proposed = _power_of_two_scale(proposed)
    updated_step = step + 1
    update = seen & (updated_step.remainder(interval) == 0)
    return torch.where(update, proposed, scales), updated_history, torch.zeros_like(observed), updated_step


@_delayed_advance.register_fake
def _advance_fake(scales, history, observed, step, interval, history_max, power_of_two,
                  input_maximum, weight_maximum, grad_maximum):
    return torch.empty_like(scales), torch.empty_like(history), torch.empty_like(observed), torch.empty_like(step)


class Float8ScaleState(nn.Module):
    """Persistent delayed scales, with idempotent microbatch/recompute observations.

    Active scales never change during forward/backward. Only advance() changes
    the history once after a complete optimizer attempt. Thus activation
    checkpoint recomputation observes the same scale and cannot advance history.
    """
    def __init__(self, recipe, *, device, group=None):
        super().__init__()
        self.recipe, self.group = recipe, group
        self.register_buffer('scales', torch.ones(3, dtype=torch.float32, device=device))
        self.register_buffer('history', torch.zeros((3, recipe.amax_history_len, 2), dtype=torch.float32, device=device))
        self.register_buffer('pending', torch.zeros((3, 3), dtype=torch.float32, device=device))
        self.register_buffer('step', torch.zeros((), dtype=torch.int64, device=device))

    def _apply(self, fn, recurse=True):
        original = dict(self.named_buffers())
        super()._apply(fn, recurse=recurse)
        for name, source in original.items():
            target = getattr(self, name)
            if source.device.type == 'meta' and target.device.type != 'meta':
                target = torch.ones(source.shape, dtype=source.dtype, device=target.device) if name == 'scales' else torch.zeros(source.shape, dtype=source.dtype, device=target.device)
            else:
                target = source.to(device=target.device, dtype=source.dtype)
            setattr(self, name, target)
        return self

    def quantize(self, value, role, *, observe=True):
        dtype = (self.recipe.input_dtype, self.recipe.weight_dtype, self.recipe.grad_dtype)[role]
        if not observe:
            scale = self.scales[role].clone()
            return _encode(value.detach().contiguous(), scale, dtype), scale
        return _delayed_encode(value.detach().contiguous(), self.scales[role], self.pending[role], dtype)

    @torch.no_grad()
    def advance(self, *, group=None):
        groups = []
        for peer_group in (self.group, group):
            if peer_group is not None and not any(peer_group is existing for existing in groups):
                groups.append(peer_group)
        observed = self.pending.clone()
        from ._collective_ops import collective_reduce, group_key
        for peer_group in groups:
            observed = collective_reduce(observed, 1, group_key(peer_group))
        maxima = [torch.finfo(dtype).max / (2. ** self.recipe.margin)
                  for dtype in (self.recipe.input_dtype, self.recipe.weight_dtype, self.recipe.grad_dtype)]
        scales, history, pending, step = _delayed_advance(self.scales, self.history, observed, self.step,
            self.recipe.scaling_interval, self.recipe.amax_compute_algo == 'max', self.recipe.power_of_two_scale, *maxima)
        self.scales.copy_(scales)
        self.history.copy_(history)
        self.pending.copy_(pending)
        self.step.copy_(step)


def make_scale_state(recipe, weight, group=None, source=None):
    if group is not None:
        from ._collective_ops import group_key
        group_key(group)
    if recipe is None or recipe.scaling != 'delayed':
        return None
    state = Float8ScaleState(recipe, device=weight.device, group=group)
    if isinstance(source, Float8ScaleState) and source.recipe == recipe:
        state.load_state_dict(source.state_dict())
    return state


def iter_float8_states(model):
    """Bind FSDP template scale objects to their persistent local buffer storage."""
    from .sharded_training import FullyShardedModule
    for path, module in model.named_modules():
        if isinstance(module, Float8ScaleState):
            yield path, module
        elif isinstance(module, FullyShardedModule):
            buffers = dict(module._buffer_names)
            for child_path, child in module._template.named_modules():
                if not isinstance(child, Float8ScaleState):
                    continue
                view = copy.copy(child)
                view._buffers = dict(child._buffers)
                for name in child._buffers:
                    key = buffers[(child_path + '.' if child_path else '') + name]
                    if key in module._buffer_shards:
                        raise ValueError('delayed FP8 histories are mutable replica buffers, not sharded read-only buffers')
                    view._buffers[name] = module.get_buffer(key)
                view.group = child.group if child.group is not None else module.group
                yield '.'.join(part for part in (path, child_path) if part), view


@torch.no_grad()
def advance_float8_scales(model, *, group=None):
    """Commit accumulated delayed observations once per optimizer attempt.

    Call after backward, also after a skipped optimizer update. SFTTrainer calls
    this automatically; custom training loops call it at the same boundary.
    State is ordinary persistent model buffers and is restored by state_dict.
    """
    if group is not None and hasattr(group, 'advance_float8_scales'):
        return group.advance_float8_scales(model)
    states = [state for _, state in iter_float8_states(model)]
    if not states:
        return
    if group is not None and not hasattr(group, 'max_'):
        group = getattr(group, 'data', getattr(group, 'transport', group))
        if not hasattr(group, 'max_'):
            raise ValueError('delayed FP8 synchronization requires an actual replica communicator')
    for state in states:
        state.advance(group=group)


class _RecipeLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, recipe, state, group, axis, reduce, observe, replicas):
        if recipe.scaling == 'block':
            output = _block_product(value, weight, recipe, 0, 1, group, 1 if axis == 1 else -1, axis, replicas)
            ctx.save_for_backward(value, weight)
        else:
            qx, sx = state.quantize(value, 0, observe=observe)
            qw, sw = state.quantize(weight, 1, observe=observe)
            output = _mm(qx, qw, sx, sw, False, True, _STRATEGIES[recipe.strategy])
            ctx.save_for_backward(qx, qw, sx, sw, state.scales[2].clone())
            ctx.gradient_pending = state.pending[2]
        if reduce and axis == 1:
            from ._collective_ops import collective_reduce, group_key
            output = collective_reduce(output, 0, group_key(group))
        ctx.recipe, ctx.state, ctx.group, ctx.axis, ctx.reduce = recipe, state, group, axis, reduce
        ctx.observe = observe
        ctx.replicas = replicas
        ctx.input_dtype, ctx.weight_dtype = value.dtype, weight.dtype
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        if gradient is None:
            return (None,) * 9
        recipe, dx, dw = ctx.recipe, None, None
        if recipe.scaling == 'block':
            value, weight = ctx.saved_tensors
            if ctx.needs_input_grad[0]:
                dx = _block_product(gradient, weight.t(), recipe, 2, 1, ctx.group,
                    1 if ctx.axis == 0 else -1, 1 - ctx.axis if ctx.axis >= 0 else -1, ctx.replicas)
            if ctx.needs_input_grad[1]:
                dw = _block_product(gradient.t(), value.t(), recipe, 2, 0, ctx.group,
                    0 if ctx.axis == 0 else -1, 0 if ctx.axis == 1 else -1, ctx.replicas)
        else:
            qx, qw, sx, sw, grad_scale = ctx.saved_tensors
            if ctx.observe:
                qg, sg = _delayed_encode(gradient.detach().contiguous(), grad_scale, ctx.gradient_pending, recipe.grad_dtype)
            else:
                qg, sg = _encode(gradient.detach().contiguous(), grad_scale, recipe.grad_dtype), grad_scale
            if ctx.needs_input_grad[0]:
                dx = _mm(qg, qw, sg, sw, False, False, _STRATEGIES[recipe.strategy])
            if ctx.needs_input_grad[1]:
                dw = _mm(qg, qx, sg, sx, True, False, _STRATEGIES[recipe.strategy])
        if dx is not None:
            if ctx.reduce and ctx.axis == 0:
                from ._collective_ops import collective_reduce, group_key
                dx = collective_reduce(dx, 0, group_key(ctx.group))
            dx = dx.to(ctx.input_dtype)
        if dw is not None:
            dw = dw.to(ctx.weight_dtype)
        return dx, dw, None, None, None, None, None, None, None


def recipe_linear(value, weight, recipe, state=None, *, group=None, axis=-1, reduce=False, observe=True, replicas=1):
    if recipe.scaling == 'delayed' and (not isinstance(state, Float8ScaleState) or state.recipe != recipe):
        raise ValueError('delayed FP8 requires the matching persistent Float8ScaleState')
    return _RecipeLinear.apply(value, weight, recipe, state, group, axis, reduce, observe and torch.is_grad_enabled(), replicas)
