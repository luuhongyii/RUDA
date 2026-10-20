"""Document-preserving response layout operations for post-training."""
from __future__ import annotations

import torch


def response_fields(packed=False):
    return {'input_ids', 'attention_mask', 'labels'} | (
        {'position_ids', 'cu_seqlens', 'max_seqlen'} if packed else set())


def batch_metadata(input_ids, metadata, packed):
    expected = {'position_ids', 'cu_seqlens', 'max_seqlen'} if packed else set()
    if set(metadata) != expected:
        raise ValueError('response metadata must match the padded/packed model layout')
    if not packed:
        if input_ids.ndim != 2:
            raise ValueError('padded tokens must have [B,T] layout')
        return metadata
    from .varlen_attention import _boundaries
    if input_ids.ndim != 1:
        raise ValueError('packed tokens must have a flat [N] layout')
    boundaries = _boundaries(metadata['cu_seqlens'], input_ids.numel())
    lengths = [b-a for a, b in zip(boundaries, boundaries[1:])]
    positions = metadata['position_ids']
    if (min(lengths) < 1 or type(metadata['max_seqlen']) is not int
            or metadata['max_seqlen'] != max(lengths)):
        raise ValueError('max_seqlen must match nonempty packed documents')
    if (not isinstance(positions, torch.Tensor) or positions.shape != input_ids.shape
            or positions.device != input_ids.device or positions.dtype not in (torch.int32, torch.int64)
            or (positions < 0).any()):
        raise ValueError('packed position_ids must be corresponding nonnegative integer tokens')
    return metadata


def response_spans(batch):
    if 'cu_seqlens' in batch:
        boundaries = batch['cu_seqlens'].tolist()
        return list(zip(boundaries, boundaries[1:]))
    return [(0, int(length)) for length in batch['attention_mask'].sum(-1)]


def response_count(batch):
    return batch['cu_seqlens'].numel()-1 if 'cu_seqlens' in batch else batch['input_ids'].shape[0]


def response_batch(batch, *, cpu=False):
    ids, mask, labels = (batch[k] for k in ('input_ids', 'attention_mask', 'labels'))
    packed = 'cu_seqlens' in batch
    metadata = {k: batch[k] for k in ('position_ids', 'cu_seqlens', 'max_seqlen') if k in batch}
    batch_metadata(ids, metadata, packed)
    if ids.shape != mask.shape or ids.shape != labels.shape or not ids.numel():
        raise ValueError('responses need matching nonempty IDs, attention mask and labels')
    if ids.dtype not in (torch.int32, torch.int64) or labels.dtype not in (torch.int32, torch.int64) or mask.dtype != torch.bool:
        raise ValueError('IDs/labels must be integers and attention_mask boolean')
    if any(t.device != ids.device for t in (mask, labels)) or cpu and ids.device.type != 'cpu':
        raise ValueError('response fields must share a device; trainer batches must be on CPU')
    supervised = labels != -100
    if (ids < 0).any() or ((labels != ids) & supervised).any() or (supervised & ~mask).any():
        raise ValueError('completion labels must equal input IDs; prompt/padding labels must be -100')
    if packed:
        if not mask.all():
            raise ValueError('packed responses contain no padding')
        for begin, end in response_spans(batch):
            active = supervised[begin:end]
            if end-begin < 2 or active[0] or not active.any() or (active[:-1] & ~active[1:]).any():
                raise ValueError('each document needs a nonempty prompt and contiguous completion suffix')
        return supervised
    if ids.shape[1] < 2 or (~mask[:, 0]).any() or supervised[:, 0].any() or ((~mask[:, :-1]) & mask[:, 1:]).any():
        raise ValueError('responses require a nonempty prompt and right padding')
    if ((supervised[:, :-1] & mask[:, 1:]) & ~supervised[:, 1:]).any() or (supervised[:, 1:].sum(-1) == 0).any():
        raise ValueError('each response must have a nonempty contiguous completion suffix')
    return supervised[:, 1:]


def prompt(batch, row):
    if 'cu_seqlens' in batch:
        begin, end = response_spans(batch)[row]
        labels, ids = batch['labels'][begin:end], batch['input_ids'][begin:end]
    else:
        labels, ids = batch['labels'][row], batch['input_ids'][row]
    first = int((labels != -100).nonzero()[0, 0])
    return ids[:first].tolist()


def document_sum(tokens, batch):
    if 'cu_seqlens' not in batch:
        return tokens.sum(-1)
    return torch.stack([tokens[a:b].sum() for a, b in response_spans(batch)])


def action_values(values, batch):
    if 'cu_seqlens' not in batch:
        return values[:, :-1]
    return torch.cat([torch.cat((values.new_zeros(1), values[a:b-1]))
                      for a, b in response_spans(batch)])


def prefix_metadata(batch, begin, end):
    return dict(position_ids=batch['position_ids'][begin:end],
                cu_seqlens=batch['cu_seqlens'].new_tensor([0, end-begin]), max_seqlen=end-begin)


def move_batch(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) and k != 'cu_seqlens' else v
            for k, v in batch.items()}


def clone_batch(batch):
    return {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def select_response_batch(batch, indices):
    """Select whole responses, preserving fixed rollout statistics and positions.

    Preference indices must retain chosen-then-rejected pairing. GRPO groups
    must be snapshotted in full before selection. Packed token statistics are
    target-aligned [N]; response-level GRPO advantages are [documents].
    """
    if 'response_metadata' in batch:
        from ._context_post_training import select_context_batch
        return select_context_batch(batch, indices)
    response_batch(batch)
    if isinstance(indices, torch.Tensor):
        if indices.device.type != 'cpu' or indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError('selection indices must be a CPU integer vector')
        indices = indices.tolist()
    else:
        indices = list(indices)
    n = response_count(batch)
    if not indices or any(type(i) is not int or not 0 <= i < n for i in indices):
        raise ValueError('select nonempty in-range response indices')
    packed = 'cu_seqlens' in batch
    rollout_fields = {'old_log_probs', 'old_values', 'reference_log_probs', 'advantages', 'returns'}
    if set(batch)-response_fields(packed)-rollout_fields:
        raise ValueError('unknown response/rollout selection fields')
    if not packed:
        return {k: v[indices].clone() for k, v in batch.items()}
    documents = response_spans(batch)
    spans = [documents[i] for i in indices]
    lengths = [b-a for a, b in spans]
    result = {}
    for name, value in batch.items():
        if name in ('cu_seqlens', 'max_seqlen'):
            continue
        if name == 'advantages' and value.shape == (n,):
            result[name] = value[indices].clone()
        elif value.shape == batch['input_ids'].shape:
            result[name] = torch.cat([value[a:b] for a, b in spans])
        else:
            raise ValueError(f'invalid packed token field: {name}')
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1]+length)
    result.update(cu_seqlens=batch['cu_seqlens'].new_tensor(boundaries), max_seqlen=max(lengths))
    return result
