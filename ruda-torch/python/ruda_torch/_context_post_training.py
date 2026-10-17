"""Context-sharded post-training; hidden states and attention KV remain local."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .causal_finetuning import ContextParallelFinetuner
from ._post_training_batch import (response_batch, response_count, response_fields,
    response_spans, clone_batch, move_batch)
from ._collective_ops import group_key


CONTEXT_FIELDS = {'input_ids', 'attention_mask', 'position_ids', 'ruda_context_documents',
                  'context_targets', 'response_metadata'}
STAT_FIELDS = {'old_log_probs', 'reference_log_probs', 'old_values', 'advantages', 'returns'}


def _source_lanes(source):
    if 'cu_seqlens' in source:
        ids = source['input_ids'].unsqueeze(0)
        labels = source['labels'].unsqueeze(0)
        positions = source['position_ids'].long().unsqueeze(0)
        documents = torch.empty_like(ids, dtype=torch.int64)
        for row, (a, b) in enumerate(response_spans(source)):
            documents[:, a:b] = row
        return ids, labels, source['attention_mask'].unsqueeze(0), positions, documents
    ids = source['input_ids']
    positions = torch.arange(ids.shape[1], dtype=torch.int64).expand_as(ids)
    documents = torch.arange(ids.shape[0], dtype=torch.int64)[:, None].expand_as(ids)
    return ids, source['labels'], source['attention_mask'], positions, documents


def _shard(source, rank, world, *, copy_source=True):
    response_batch(source, cpu=True)
    if type(rank) is not int or type(world) is not int or not 0 <= rank < world:
        raise ValueError('invalid context rank/world size')
    fields = response_fields('cu_seqlens' in source)
    if set(source)-fields-STAT_FIELDS:
        raise ValueError('unknown response/rollout fields')
    if any(value.requires_grad or value.device.type != 'cpu' for value in source.values()
           if isinstance(value, torch.Tensor)):
        raise ValueError('context sources must be detached CPU response/rollout batches')
    ids, labels, mask, positions, documents = _source_lanes(source)
    width = ids.shape[1]
    begin, end = width*rank//world, width*(rank+1)//world
    targets = torch.full_like(labels, -100)
    valid = mask[:, :-1] & mask[:, 1:] & (documents[:, :-1] == documents[:, 1:])
    targets[:, :-1] = torch.where(valid, labels[:, 1:], -100)
    result = {k: value[:, begin:end].contiguous().clone() for k, value in
              dict(input_ids=ids, attention_mask=mask, position_ids=positions,
                   ruda_context_documents=documents, context_targets=targets).items()}
    result['response_metadata'] = dict(source=clone_batch(source) if copy_source else source, rank=rank, world_size=world,
                                       start=begin, end=end, width=width)
    for name in set(source)-fields:
        value = source[name]
        if name == 'advantages' and value.shape == (response_count(source),):
            result[name] = value.clone()
            continue
        expected = response_batch(source).shape
        if value.shape != expected or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f'invalid finite floating rollout field: {name}')
        # Global sources use target alignment; CP losses use the local state
        # that predicts the next token, shifted before the context partition.
        aligned = torch.cat((value[1:], value.new_zeros(1))).unsqueeze(0) if 'cu_seqlens' in source else F.pad(value, (0, 1))
        result[name] = aligned[:, begin:end].contiguous().clone()
    return result


def shard_post_training_batch(batch, group):
    """Shard a complete CPU padded/packed response or fixed rollout over CP.

    The nested CPU source retains token identity and fixed scalar statistics,
    not activations/KV/weights. It also permits document-preserving selection
    and exact step-boundary continuation without rescoring the old policy.
    """
    return _shard(batch, group.rank, group.world_size)


def select_context_batch(batch, indices):
    from ._post_training_batch import select_response_batch
    meta = batch['response_metadata']
    return _shard(select_response_batch(meta['source'], indices), meta['rank'], meta['world_size'])


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def validate_context_batch(batch, group, *, cpu=False, collective=False):
    if not CONTEXT_FIELDS <= set(batch) or set(batch)-CONTEXT_FIELDS-STAT_FIELDS:
        raise ValueError('supply a context response batch produced by shard_post_training_batch')
    meta = batch['response_metadata']
    if not isinstance(meta, dict) or set(meta) != {'source', 'rank', 'world_size', 'start', 'end', 'width'}:
        raise ValueError('invalid context response metadata')
    if (meta['rank'], meta['world_size']) != (group.rank, group.world_size):
        raise ValueError('context response belongs to a different rank/topology')
    expected = _shard(meta['source'], group.rank, group.world_size, copy_source=False)
    if any(meta[k] != expected['response_metadata'][k] for k in ('start', 'end', 'width')):
        raise ValueError('context response range differs from its actual source')
    ids = batch['input_ids']
    for name in CONTEXT_FIELDS-{'response_metadata'}:
        value = batch[name]
        if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype or value.device != ids.device
                or cpu and value.device.type != 'cpu' or value.requires_grad):
            raise ValueError('invalid context token/target tensor')
        if cpu and not torch.equal(value, expected[name]):
            raise ValueError('context token/target fields differ from their source')
    for name in set(batch)-CONTEXT_FIELDS:
        value = batch[name]
        if (name not in expected or value.shape != expected[name].shape or not value.is_floating_point()
                or value.device != ids.device or value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError(f'invalid finite floating context rollout field: {name}')
        if cpu and not torch.equal(value, expected[name]):
            raise ValueError('fixed context rollout statistics differ from their source')
    if collective:
        group.validate_training_options(_plain(meta['source']))
    return meta['source']


def _hidden(policy, batch):
    from .post_training import _training_hidden
    return _training_hidden(policy, batch['input_ids'], batch['attention_mask'],
        position_ids=batch['position_ids'], ruda_context_positions=batch['position_ids'],
        ruda_context_documents=batch['ruda_context_documents'], ruda_context_valid=batch['attention_mask'])


class ContextParallelValueModel(ContextParallelFinetuner):
    """Independent scalar-head path using the actual context-parallel backbone."""
    def forward(self, input_ids, attention_mask, position_ids, ruda_context_documents):
        values = self.head(_hidden(self, dict(input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, ruda_context_documents=ruda_context_documents)))
        if values.shape != (*input_ids.shape, 1):
            raise ValueError('context critic head must produce one scalar per local token')
        return values.squeeze(-1).float()


def _sum(value, group):
    from .transformers_context_parallel import _sum_objective
    return _sum_objective(value, group_key(group))


@torch.library.custom_op('ruda::context_response_transfer', mutates_args=())
def _transfer(value: torch.Tensor, key: str) -> torch.Tensor:
    from ._expert_exchange import _group
    return _group(key).sum_(value.detach().clone())


@_transfer.register_fake
def _transfer_fake(value, key):
    return torch.empty_like(value)


def _transfer_setup(ctx, inputs, output):
    ctx.key = inputs[1]


def _transfer_backward(ctx, gradient):
    # A prefix's last state and its target in the full response can belong to
    # different ranks. Both primal and adjoint transfer between those owners.
    return _transfer(gradient, ctx.key), None


_transfer.register_autograd(_transfer_backward, setup_context=_transfer_setup)


def _documents(source):
    ids, labels, _, positions, _ = _source_lanes(source)
    packed = 'cu_seqlens' in source
    for row, (a, b) in enumerate(response_spans(source)):
        lane = 0 if packed else row
        first = int((labels[lane, a:b] != -100).nonzero()[0, 0])
        yield row, lane, a, b, first, ids[lane, a:b], positions[lane, a:b]


def _prefix(policy, tokens, positions, length, device):
    group = policy.context_group
    a, b = length*group.rank//group.world_size, length*(group.rank+1)//group.world_size
    ids = tokens[:length].unsqueeze(0)
    batch = move_batch(dict(input_ids=ids[:, a:b], attention_mask=torch.ones_like(ids[:, a:b], dtype=torch.bool),
        position_ids=positions[:length].unsqueeze(0)[:, a:b],
        ruda_context_documents=torch.zeros_like(ids[:, a:b])), device)
    hidden = _hidden(policy, batch)
    return hidden, a <= length-1 < b


def _scalar_values(policy, batch):
    return policy(batch['input_ids'], batch['attention_mask'], batch['position_ids'],
                  batch['ruda_context_documents'])


def _prefix_statistics(model, policy, batch, *, values=False):
    from .post_training import chunked_lm_log_probs
    source = batch['response_metadata']['source']
    start, end = batch['response_metadata']['start'], batch['response_metadata']['end']
    device = batch['input_ids'].device
    scores, slots = [], []
    for _, lane, a, b, first, tokens, positions in _documents(source):
        steps = range(first-1, b-a) if values else range(first, b-a)
        for t in steps:
            length = t+1 if values else t
            hidden, owner = _prefix(policy, tokens, positions, length, device)
            if owner:
                if values:
                    local = policy.head(hidden[:, -1:]).reshape(()).float()
                else:
                    target = tokens[t:t+1].to(device).reshape(1, 1)
                    local = chunked_lm_log_probs(hidden[:, -1:], policy.head, target,
                                                shift=False, token_chunk_size=1).reshape(())
            else:
                local = hidden.sum().float()*0
            score = _transfer(local, group_key(model.context_group))
            state = a+t if values else a+t-1
            slot = lane*(end-start)+state-start if start <= state < end else None
            scores.append(score if slot is not None else score*0)
            slots.append(0 if slot is None else slot)
    scores = torch.stack(scores)
    result = scores.new_zeros(batch['input_ids'].shape)
    if result.numel():
        index = torch.tensor(slots, device=device, dtype=torch.long)
        result = result.flatten().scatter_add(0, index, scores).reshape(result.shape)
    else:
        result = result+scores.sum()*0
    return result


def _logp(model, policy, batch):
    from .post_training import chunked_lm_log_probs
    if model.prefix_scoring:
        return _prefix_statistics(model, policy, batch)
    return chunked_lm_log_probs(_hidden(policy, batch), policy.head, batch['context_targets'],
                                token_chunk_size=policy.token_chunk_size, shift=False)


def _values(model, batch):
    return (_prefix_statistics(model, model.critic, batch, values=True) if model.prefix_scoring
            else _scalar_values(model.critic, batch))


def _document_sum(local, batch, group):
    source = batch['response_metadata']['source']
    documents = batch['ruda_context_documents'].reshape(-1)
    sums = local.new_zeros(response_count(source)).scatter_add(0, documents, local.reshape(-1))
    return _sum(sums, group)


def _reduce_replicated(losses, reduction, group):
    if reduction == 'mean':
        return losses.mean()
    if reduction == 'sum':
        loss = losses.sum()
        # Report this replicated objective once, while retaining its full local
        # derivative; the coordinator subsequently sums disjoint CP gradients.
        return loss+(loss.detach()/group.world_size-loss.detach())
    raise ValueError('objective reduction must be sum or mean')


def context_forward(model, batch, *, mode, reduction, **statistics):
    from .post_training import _preference_batch
    source = validate_context_batch(batch, model.context_group, collective=True)
    algorithm = model.contract()['algorithm']
    group = model.context_group
    if algorithm == 'reward':
        n = _preference_batch(source) if mode == 'loss' else None
        if mode not in ('loss', 'score'):
            raise ValueError('unknown reward forward mode')
        if model.prefix_scoring:
            scores = []
            for _, _, _, b, _, tokens, positions in _documents(source):
                hidden, owner = _prefix(model.policy, tokens, positions, len(tokens), batch['input_ids'].device)
                local = model.policy.head(hidden[:, -1:]).reshape(()).float() if owner else hidden.sum().float()*0
                scores.append(_sum(local, group))
            scores = torch.stack(scores)
        else:
            values = _scalar_values(model.policy, batch)
            start, end = batch['response_metadata']['start'], batch['response_metadata']['end']
            selected = []
            for _, lane, _, b, *_ in _documents(source):
                selected.append(values[lane, b-1-start] if start <= b-1 < end else values.sum()*0)
            scores = _sum(torch.stack(selected), group)
        return scores if mode == 'score' else _reduce_replicated(-F.logsigmoid(scores[:n]-scores[n:]), reduction, group)
    mask = batch['context_targets'] != -100
    logp = _logp(model, model.policy, batch)
    if algorithm == 'dpo':
        if mode != 'loss':
            raise ValueError('unknown DPO forward mode')
        n = _preference_batch(source)
        with torch.no_grad():
            reference = _logp(model, model.reference, batch)
        score = _document_sum(logp-reference, batch, group)
        return _reduce_replicated(-F.logsigmoid(model.beta*(score[:n]-score[n:])), reduction, group)
    values = _values(model, batch) if algorithm == 'ppo' else None
    if mode == 'snapshot':
        with torch.no_grad():
            ref = _logp(model, model.reference, batch)
        return (logp.detach(), values.detach(), ref.detach()) if algorithm == 'ppo' else (logp.detach(), ref.detach())
    if mode != 'loss':
        raise ValueError('unknown post-training forward mode')
    ratio = (logp-statistics['old_log_probs'].detach()).exp()
    advantages = statistics['advantages'].detach()
    if algorithm == 'grpo':
        advantages = advantages[batch['ruda_context_documents']]
    policy = torch.minimum(ratio*advantages, ratio.clamp(1-model.clip_ratio, 1+model.clip_ratio)*advantages)
    if algorithm == 'ppo':
        returns, old = statistics['returns'].detach(), statistics['old_values'].detach()
        value_loss = (values-returns).square()
        if model.value_clip is not None:
            clipped = old+(values-old).clamp(-model.value_clip, model.value_clip)
            value_loss = torch.maximum(value_loss, (clipped-returns).square())
        loss = torch.where(mask, -policy+.5*model.value_coefficient*value_loss, 0.).sum()
        if reduction == 'sum':
            return loss
        if reduction != 'mean':
            raise ValueError('objective reduction must be sum or mean')
        return _sum(loss, group)/_sum(mask.sum().float(), group)
    delta = statistics['reference_log_probs'].detach()-logp
    local = torch.where(mask, model.kl_coefficient*(delta.exp()-delta-1)-policy, 0.)
    means = _document_sum(local, batch, group)/_document_sum(mask.float(), batch, group)
    return _reduce_replicated(means, reduction, group)


@torch.no_grad()
def gather_context_statistics(local, batch, group, *, action=True):
    """Gather only detached scalar rollout statistics, not hidden states or KV."""
    meta = batch['response_metadata']
    width, world = meta['width'], group.world_size
    maximum = (width+world-1)//world
    padded = F.pad(local.detach(), (0, maximum-local.shape[1]))
    gathered = group.all_gather(padded.contiguous(), axis=1)
    complete = torch.cat([gathered[:, rank*maximum:rank*maximum+width*(rank+1)//world-width*rank//world]
                          for rank in range(world)], 1).cpu()
    if 'cu_seqlens' in meta['source']:
        flat = complete.reshape(-1)
        return torch.cat((flat.new_zeros(1), flat[:-1])) if action else flat
    return complete[:, :-1] if action else complete


@torch.no_grad()
def context_sample(trainer, prompt, *, max_new_tokens, eos_token_ids, generator=None):
    from ._architecture_ops import positive_int
    positive_int(max_new_tokens, 'max_new_tokens')
    if not prompt or any(type(t) is not int or t < 0 for t in [*prompt, *eos_token_ids]):
        raise ValueError('supply nonempty prompt and nonnegative prompt/EOS token IDs')
    model = getattr(trainer.model, '_template', trainer.model)
    group = model.context_group
    group.validate_training_options((list(prompt), max_new_tokens, list(eos_token_ids)))
    device = next(trainer.model.parameters()).device
    ids, terminal = list(prompt), False
    trainer.executable.eval()
    for _ in range(max_new_tokens):
        tokens = torch.tensor(ids, dtype=torch.long, device='cpu')
        a, b = len(ids)*group.rank//group.world_size, len(ids)*(group.rank+1)//group.world_size
        batch = move_batch(dict(input_ids=tokens[a:b].unsqueeze(0),
            attention_mask=torch.ones((1, b-a), dtype=torch.bool),
            position_ids=torch.arange(len(ids))[a:b].unsqueeze(0),
            ruda_context_documents=torch.zeros((1, b-a), dtype=torch.int64)), device)
        logits = trainer.executable(**batch, context_width=len(ids), mode='next_logits')
        token = torch.multinomial(logits.float().softmax(-1).cpu(), 1, generator=generator).to(device)
        group.broadcast_(token, root=0)
        if trainer.sampling_group is not None:
            trainer.sampling_group.broadcast_(token, root=0)
        ids.append(int(token.cpu()[0, 0]))
        if ids[-1] in eos_token_ids:
            terminal = True
            break
    tokens = torch.tensor([ids], dtype=torch.long, device='cpu')
    labels = tokens.clone(); labels[:, :len(prompt)] = -100
    return dict(input_ids=tokens, labels=labels, attention_mask=torch.ones_like(tokens, dtype=torch.bool)), terminal


def context_next_logits(model, input_ids, attention_mask, position_ids, ruda_context_documents, context_width):
    from .post_training import _parallel_head
    group = model.context_group
    if type(context_width) is not int or context_width < 1:
        raise ValueError('context sampling requires a positive complete-prefix width')
    length = context_width*(group.rank+1)//group.world_size-context_width*group.rank//group.world_size
    if input_ids.shape != (1, length):
        raise ValueError('context sampling requires an actual single-prefix shard')
    hidden = _hidden(model.policy, dict(input_ids=input_ids, attention_mask=attention_mask,
        position_ids=position_ids, ruda_context_documents=ruda_context_documents))
    owner = group.rank == group.world_size-1
    last = hidden[:, -1:] if owner else hidden.new_zeros((1, 1, hidden.shape[-1]))
    logits = model.policy.head(last).reshape(1, -1)
    if _parallel_head(model.policy.head) and not model.policy.head.gather_output:
        from .parallel_training import gather_from_tensor_parallel
        logits = gather_from_tensor_parallel(logits, model.policy.head.group)
    return _sum(logits if owner else logits*0, group)
