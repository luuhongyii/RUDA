"""Packed post-training against independent padded causal-attention objectives."""
import copy
import importlib
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch import nn
from torch.nn import functional as F
from architecture_test_utils import NAME

pt = importlib.import_module(NAME+'.post_training')
cft = importlib.import_module(NAME+'.causal_finetuning')
ft = importlib.import_module(NAME+'.finetuning')
data_ops = importlib.import_module(NAME+'.training_data')
varlen = importlib.import_module(NAME+'.varlen_attention')
torch.set_num_threads(1)


class AttentionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 6)
        self.positions = nn.Embedding(32, 6)
        self.qkv = nn.Linear(6, 18, bias=False)
        self.output = nn.Linear(6, 6, bias=False)
        self.dropout = nn.Dropout(.4)
        self.length_scale = 0.

    def forward(self, input_ids, attention_mask, position_ids=None, cu_seqlens=None, max_seqlen=None):
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device)
        x = self.embedding(input_ids)+self.positions(position_ids)
        q, k, v = self.qkv(x).chunk(3, -1)
        if cu_seqlens is not None:
            attended = varlen.varlen_scaled_dot_product_attention(q[:, None], k[:, None], v[:, None],
                cu_seqlens, is_causal=True, query_block_size=2, key_block_size=3).squeeze(1)
            length = max_seqlen
        else:
            # Independent torch SDPA oracle; not the production varlen operation.
            causal = torch.ones(input_ids.shape[-1], input_ids.shape[-1], dtype=torch.bool).tril()
            mask = causal[None, None] & attention_mask[:, None, None, :]
            attended = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None],
                                                       attn_mask=mask).squeeze(1)
            length = input_ids.shape[-1]
        return self.dropout(x+self.output(attended))*attention_mask[..., None]*(1+self.length_scale*length)


def samples():
    ids = [[1, 2, 3, 4, 5], [1, 2, 6], [3, 4, 7, 8], [3, 4, 9, 10, 11, 12]]
    return [dict(input_ids=tokens, labels=[-100, -100]+tokens[2:]) for tokens in ids]


def responses(packed):
    return (data_ops.collate_varlen_causal_lm(samples()) if packed else
            cft.SFTCollator(max_length=12, pad_token_id=0)(samples()))


def make(algorithm, packed=True, prefix=False):
    torch.manual_seed(501)
    cls = cft.PackedCausalLMFinetuner if packed else cft.CausalLMFinetuner
    actor = cls(AttentionBackbone(), nn.Linear(6, 13), token_chunk_size=2)
    reference = copy.deepcopy(actor)
    with torch.no_grad():
        reference.head.weight.add_(torch.randn_like(reference.head.weight)*.1)
    value_cls = pt.PackedCausalValueModel if packed else pt.CausalValueModel
    if algorithm == 'reward':
        model = pt.RewardModel(value_cls(AttentionBackbone(), nn.Linear(6, 1, bias=False)), prefix_scoring=prefix)
        trainer_cls = pt.RewardTrainer
    elif algorithm == 'dpo':
        model = pt.DPOModel(actor, reference, reference_id='packed-reference', beta=.4, prefix_scoring=prefix)
        trainer_cls = pt.DPOTrainer
    elif algorithm == 'ppo':
        critic = value_cls(AttentionBackbone(), nn.Linear(6, 1, bias=False))
        model = pt.PPOModel(actor, reference, critic, reference_id='packed-reference', kl_coefficient=.1,
                            gamma=.9, gae_lambda=.7, prefix_scoring=prefix)
        trainer_cls = pt.PPOTrainer
    else:
        model = pt.GRPOModel(actor, reference, reference_id='packed-reference', prefix_scoring=prefix)
        trainer_cls = pt.GRPOTrainer
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01, momentum=.9)
    return model, trainer_cls(model, optimizer, base_id='packed-base', run_config={'source': 'exact'})


def prepare(algorithm, trainer, packed):
    batch = responses(packed)
    if algorithm in ('dpo', 'reward'):
        return pt.select_response_batch(batch, [0, 2, 1, 3])
    rewards = torch.tensor([1., -.5, .3, 2.])
    if algorithm == 'grpo':
        return trainer.prepare_groups(batch, rewards, torch.tensor([0, 0, 1, 1]))
    token_rewards = torch.zeros_like(pt._response_batch(batch), dtype=torch.float32)
    if packed:
        token_rewards[batch['cu_seqlens'][1:]-1] = rewards
    else:
        token_rewards[torch.arange(4), batch['attention_mask'].sum(-1)-2] = rewards
    return trainer.prepare_rollout(batch, token_rewards, torch.tensor([True, False, True, False]))


def packed_from_padded_stat(stat, packed):
    if stat.ndim == 1:
        return stat
    result = []
    for row, (a, b) in enumerate(pt.response_spans(packed)):
        result.append(torch.cat((stat.new_zeros(1), stat[row, :b-a-1])))
    return torch.cat(result)


@pytest.mark.parametrize('algorithm', ['reward', 'dpo', 'ppo', 'grpo'])
@pytest.mark.parametrize('prefix', [False, True])
def test_packed_losses_gradients_and_snapshots_match_independent_attention(algorithm, prefix):
    packed_model, packed_trainer = make(algorithm, prefix=prefix)
    padded_model, padded_trainer = make(algorithm, packed=False, prefix=prefix)
    if prefix:
        for model in (packed_model, padded_model):
            for module in model.modules():
                if isinstance(module, AttentionBackbone):
                    module.length_scale = .1
    packed = prepare(algorithm, packed_trainer, True)
    padded = prepare(algorithm, padded_trainer, False)
    for field in ('old_log_probs', 'old_values', 'reference_log_probs', 'advantages', 'returns'):
        if field in packed:
            expected = packed_from_padded_stat(padded[field], packed)
            if field == 'old_values':
                # Values outside completion targets do not participate in PPO.
                mask = pt._response_batch(packed)
                torch.testing.assert_close(packed[field][mask], expected[mask])
            else:
                torch.testing.assert_close(packed[field], expected)
    if algorithm in ('ppo', 'grpo'):
        # Exercise clipping rather than comparing only ratio-one objectives.
        perturbation = torch.linspace(-.7, .7, padded['old_log_probs'].numel()).reshape(padded['old_log_probs'].shape)
        padded['old_log_probs'] += perturbation
        packed['old_log_probs'] += packed_from_padded_stat(perturbation, packed)
    actual, expected = packed_model(**packed), padded_model(**padded)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    actual.backward(); expected.backward()
    for (name, p), (_, q) in zip(packed_model.named_parameters(), padded_model.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-5, msg=name)


@pytest.mark.parametrize('kind', ['dense', 'lora', 'nf4', 'qlora'])
def test_flat_chunked_logp_and_gradients(kind):
    torch.manual_seed(503)
    head = nn.Linear(6, 13)
    if kind in ('nf4', 'qlora'):
        head = ft.NF4Linear.from_linear(head, block_size=8)
    if kind in ('lora', 'qlora'):
        head = ft.LoRALinear(head, rank=2, alpha=3.)
    oracle = copy.deepcopy(head)
    x = torch.randn(18, 6, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    labels = responses(True)['labels']
    actual = pt.chunked_lm_log_probs(x, head, labels, token_chunk_size=3)
    expected = F.log_softmax(oracle(y[:-1]), -1).gather(-1, labels[1:].clamp_min(0)[:, None]).squeeze(-1)
    expected = torch.where(labels[1:] != -100, expected, torch.zeros_like(expected))
    upstream = torch.randn_like(actual)
    (actual*upstream).sum().backward(); (expected*upstream).sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, y.grad)
    for p, q in zip(head.parameters(), oracle.parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize('algorithm', ['reward', 'dpo', 'ppo', 'grpo'])
def test_packed_microbatches_updates_evaluation_and_exact_resume(algorithm, tmp_path):
    model, trainer = make(algorithm)
    data = prepare(algorithm, trainer, True)
    oracle = copy.deepcopy(model)
    optimizer = torch.optim.SGD([p for p in oracle.parameters() if p.requires_grad], lr=.01, momentum=.9)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    oracle.train(); oracle(**data).backward(); optimizer.step()
    if algorithm in ('reward', 'dpo'):
        # Unequal pair counts in the two microbatches.
        data = pt.select_response_batch(data, [0, 1, 0, 2, 3, 2])
        oracle.load_state_dict(model.state_dict())
        optimizer = torch.optim.SGD([p for p in oracle.parameters() if p.requires_grad], lr=.01, momentum=.9)
        optimizer.zero_grad(set_to_none=True); oracle(**data).backward(); optimizer.step()
        rows = ([0, 3], [1, 2, 4, 5])
    else:
        rows = ([0], [1, 2, 3])
    micros = [pt.select_response_batch(data, indices) for indices in rows]
    expected_mean = float(model(**data).detach())
    evaluated = trainer.evaluate(micros)
    assert evaluated['objective_mean'] == pytest.approx(expected_mean, abs=2e-6)
    stats = trainer.train_step(micros)
    for (name, p), (_, q) in zip(model.named_parameters(), oracle.named_parameters()):
        torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5, msg=name)
        if name.startswith('reference.'):
            torch.testing.assert_close(p, before[name], atol=0, rtol=0)
    assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n.startswith('policy.'))
    if algorithm == 'ppo':
        assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n.startswith('critic.'))
    assert stats['normalization_units'] == (int(pt._response_batch(data).sum()) if algorithm == 'ppo' else
                                          pt.response_count(data) if algorithm == 'grpo' else 3)
    trainer.data_state = {'pending': data, 'epoch': 2, 'cursor': 7}
    saved = trainer.save(tmp_path)
    random = torch.rand(4)
    trainer.train_step(micros)
    expected_state = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    restored_model, restored = make(algorithm)
    assert restored.resume(saved) == 1
    torch.testing.assert_close(torch.rand(4), random, atol=0, rtol=0)
    for name, value in data.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(restored.data_state['pending'][name], value, atol=0, rtol=0)
        else:
            assert restored.data_state['pending'][name] == value
    restored.train_step(micros)
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_state[name], atol=0, rtol=0)
    for key, state in expected_optimizer['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(restored.optimizer.state_dict()['state'][key][name], value, atol=0, rtol=0)
    _, incompatible = make(algorithm, packed=False)
    with pytest.raises(ValueError, match='loss options'):
        incompatible.resume(saved)


def test_document_isolation_reward_placement_and_repacking():
    model, trainer = make('ppo')
    batch = responses(True)
    old, values, ref = model(**batch, mode='snapshot')
    changed = pt.clone_batch(batch)
    changed['input_ids'][:5] = (changed['input_ids'][:5]+1) % 13
    changed['labels'][:5] = torch.where(changed['labels'][:5] == -100, -100, changed['input_ids'][:5])
    altered = model(**changed, mode='snapshot')
    for actual, expected in zip(altered, (old, values, ref)):
        torch.testing.assert_close(actual[5:], expected[5:], atol=0, rtol=0)
    _, reward = make('reward')
    scores = reward.score(batch)
    token_rewards = reward.token_rewards(batch)
    expected = torch.zeros(18)
    expected[batch['cu_seqlens'][1:]-1] = scores
    torch.testing.assert_close(token_rewards, expected)
    fixed = trainer.prepare_rollout(batch, token_rewards, torch.tensor([True, False, True, False]))
    selected = pt.select_response_batch(fixed, [3, 0])
    assert selected['cu_seqlens'].tolist() == [0, 6, 11] and selected['max_seqlen'] == 6
    for name in ('input_ids', 'position_ids', 'old_log_probs', 'old_values', 'advantages', 'returns'):
        torch.testing.assert_close(selected[name], torch.cat((fixed[name][12:18], fixed[name][:5])), atol=0, rtol=0)
    selected['old_log_probs'].zero_()
    torch.testing.assert_close(fixed['old_log_probs'], old, atol=0, rtol=0)


@pytest.mark.parametrize('algorithm', ['dpo', 'ppo', 'grpo'])
def test_packed_sampling_uses_exact_autoregressive_prefixes_and_eos(algorithm):
    model, trainer = make(algorithm, prefix=True)
    model.policy.backbone.length_scale = .1
    seed = 508
    expected = [1, 2]
    generator = torch.Generator().manual_seed(seed)
    for _ in range(3):
        ids = torch.tensor(expected)
        metadata = dict(position_ids=torch.arange(len(expected)), cu_seqlens=torch.tensor([0, len(expected)]), max_seqlen=len(expected))
        hidden = model.policy.hidden(ids, torch.ones_like(ids, dtype=torch.bool), **metadata)
        probability = model.policy.head(hidden[-1:]).float().softmax(-1)
        expected.append(int(torch.multinomial(probability, 1, generator=generator)))
    actual, terminal = trainer.sample([1, 2], max_new_tokens=3, eos_token_ids=[],
                                      generator=torch.Generator().manual_seed(seed))
    assert not terminal and actual['input_ids'].tolist() == expected
    assert actual['cu_seqlens'].tolist() == [0, 5] and actual['position_ids'].tolist() == list(range(5))
    pt._response_batch(actual)
    actual, terminal = trainer.sample([1, 2], max_new_tokens=3, eos_token_ids=[expected[2]],
                                      generator=torch.Generator().manual_seed(seed))
    assert terminal and actual['input_ids'].tolist() == expected[:3]


def test_packed_preference_collator_and_invalid_layouts():
    rows = samples()
    collator = pt.PreferenceCollator(packed=True)
    result = collator([dict(chosen=rows[0], rejected=rows[1]), dict(chosen=rows[2], rejected=rows[3])])
    expected = pt.select_response_batch(responses(True), [0, 2, 1, 3])
    for name in result:
        if isinstance(result[name], torch.Tensor):
            torch.testing.assert_close(result[name], expected[name])
        else:
            assert result[name] == expected[name]
    with pytest.raises(ValueError, match='padding/truncation'):
        pt.PreferenceCollator(packed=True, max_length=5)
    invalid = dict(input_ids=[1, 2], labels=[1, 2])
    with pytest.raises(ValueError, match='explicitly labeled'):
        collator([dict(chosen=invalid, rejected=invalid)])
    for change in (dict(cu_seqlens=torch.tensor([0, 0, 18])), dict(max_seqlen=20),
                   dict(position_ids=torch.arange(17)), dict(labels=result['input_ids'].clone()),
                   dict(attention_mask=torch.zeros(18, dtype=torch.bool))):
        with pytest.raises(ValueError):
            pt._response_batch({**result, **change})
    with pytest.raises(ValueError, match='same response layout'):
        pt.DPOModel(make('dpo')[0].policy, make('dpo', packed=False)[0].policy, reference_id='x')
    with pytest.raises(ValueError, match='same response layout'):
        pt.PPOModel(make('ppo')[0].policy, make('ppo')[0].reference,
                    make('ppo', packed=False)[0].critic, reference_id='x')


def _distributed_worker(rank, uri, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=uri, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        replicas = importlib.import_module(NAME+'.distributed_training')
        for algorithm in ('reward', 'dpo', 'ppo', 'grpo'):
            model, original = make(algorithm)
            group = replicas.ReplicaGroup(device_type='cpu')
            group.initialize(model)
            trainer = type(original)(model, original.optimizer, base_id=original.base_id,
                                    run_config=original.run_config, replica_group=group)
            full = prepare(algorithm, trainer, True)
            if algorithm in ('reward', 'dpo'):
                full = pt.select_response_batch(full, [0, 1, 0, 2, 3, 2])
                rows = [0, 3] if rank == 0 else [1, 2, 4, 5]
            else:
                rows = [0] if rank == 0 else [1, 2, 3]
            oracle = copy.deepcopy(model)
            optimizer = torch.optim.SGD([p for p in oracle.parameters() if p.requires_grad], lr=.01, momentum=.9)
            oracle.train(); oracle(**full).backward(); optimizer.step()
            local = pt.select_response_batch(full, rows)
            trainer.train_step([local])
            for p, q in zip(model.parameters(), oracle.parameters()):
                torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)
            trainer.data_state = {'rank': rank, 'fixed_rollout': local, 'epoch': 1}
            root = Path(directory)/algorithm
            trainer.save_distributed(root)
            trainer.train_step([local])
            expected = [p.detach().clone() for p in model.parameters()]
            trainer.resume_distributed(root)
            assert trainer.step == 1 and trainer.data_state['rank'] == rank
            trainer.train_step([local])
            for p, q in zip(model.parameters(), expected):
                torch.testing.assert_close(p, q, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_packed_two_process_unequal_weighting_and_all_rank_resume(tmp_path):
    multiprocessing.spawn(_distributed_worker,
        args=((tmp_path/'rendezvous').resolve().as_uri(), str(tmp_path/'checkpoint')), nprocs=2, join=True)
