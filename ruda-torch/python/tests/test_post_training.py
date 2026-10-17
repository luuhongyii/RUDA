"""CPU numerical/update/resume regressions for the production post-training APIs."""
import copy
import importlib
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from architecture_test_utils import NAME

pt = importlib.import_module(NAME+'.post_training')
cft = importlib.import_module(NAME+'.causal_finetuning')
ft = importlib.import_module(NAME+'.finetuning')
torch.set_num_threads(1)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 5)
        self.block = nn.Sequential(nn.Linear(5, 5), nn.Tanh(), nn.Dropout(0.4))

    def forward(self, input_ids, attention_mask):
        return self.block(self.embedding(input_ids))*attention_mask.unsqueeze(-1)


def policy(kind='dense'):
    base, head = Backbone(), nn.Linear(5, 13)
    if kind in ('nf4', 'qlora'):
        head = ft.NF4Linear.from_linear(head, block_size=8, tile_rows=2)
    if kind in ('lora', 'qlora'):
        head = ft.LoRALinear(head, rank=2, alpha=3.)
    return cft.CausalLMFinetuner(base, head, token_chunk_size=2, activation_checkpointing=True)


def sample(ids, prompt=2):
    return {'input_ids': ids, 'labels': [-100]*prompt+ids[prompt:]}


def batch():
    return cft.SFTCollator(max_length=12, pad_token_id=0)([
        sample([1, 2, 3, 4, 5]), sample([1, 2, 6]),
        sample([3, 4, 7, 8]), sample([3, 4, 9, 10, 11, 12])])


@pytest.mark.parametrize('kind', ['dense', 'nf4', 'lora', 'qlora'])
@pytest.mark.parametrize('recompute', [False, True])
def test_log_probs_and_arbitrary_upstream_gradients(kind, recompute):
    torch.manual_seed(192)
    head = policy(kind).head
    oracle = copy.deepcopy(head)
    x = torch.randn(2, 6, 5, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    labels = torch.randint(0, 13, (2, 6)); labels[0, :3] = -100; labels[1, -2:] = -100
    valid = labels[:, 1:] != -100
    safe = labels[:, 1:].clamp_min(0)
    expected = F.log_softmax(oracle(xr[:, :-1]), -1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    expected = torch.where(valid, expected, torch.zeros_like(expected))
    actual = pt.chunked_lm_log_probs(x, head, labels, token_chunk_size=3, recompute=recompute)
    upstream = torch.randn_like(actual)
    (actual*upstream).sum().backward(); (expected*upstream).sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, xr.grad)
    for (name, p), (_, q) in zip(head.named_parameters(), oracle.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, msg=name)


def make(algorithm):
    torch.manual_seed(193)
    actor = policy()
    reference = copy.deepcopy(actor)
    with torch.no_grad():
        reference.head.weight.add_(torch.randn_like(reference.head.weight)*0.1)
    if algorithm == 'dpo':
        model = pt.DPOModel(actor, reference, reference_id='immutable-ref', beta=0.4)
        kind = pt.DPOTrainer
    elif algorithm == 'ppo':
        critic = pt.CausalValueModel(Backbone(), nn.Linear(5, 1), activation_checkpointing=True)
        model = pt.PPOModel(actor, reference, critic, reference_id='immutable-ref', kl_coefficient=0.1)
        kind = pt.PPOTrainer
    else:
        model = pt.GRPOModel(actor, reference, reference_id='immutable-ref')
        kind = pt.GRPOTrainer
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    return model, kind(model, optimizer, base_id='initial', run_config={'data': 'exact-input'})


def rollout(algorithm, trainer):
    data = batch()
    if algorithm == 'dpo':
        # chosen rows then rejected rows, with matching pair prompts
        order = torch.tensor([0, 2, 1, 3])
        return {k: v[order] for k, v in data.items()}
    if algorithm == 'ppo':
        rewards = torch.zeros_like(data['labels'][:, 1:], dtype=torch.float32)
        for row, reward in enumerate([1., -0.5, 0.3, 2.]):
            rewards[row, int(data['attention_mask'][row].sum())-2] = reward
        return trainer.prepare_rollout(data, rewards, torch.tensor([True, False, True, False]))
    return trainer.prepare_groups(data, torch.tensor([1., 0., 2., 2.]), torch.tensor([0, 0, 1, 1]))


def plain_logp(model, data):
    logits = model.head(model.hidden(data['input_ids'], data['attention_mask']))[:, :-1]
    targets = data['labels'][:, 1:]
    selected = F.log_softmax(logits, -1).gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return torch.where(targets != -100, selected, torch.zeros_like(selected))


@pytest.mark.parametrize('algorithm', ['dpo', 'ppo', 'grpo'])
def test_objective_matches_independent_full_projection(algorithm):
    model, trainer = make(algorithm)
    data = rollout(algorithm, trainer)
    model.train()
    logp = plain_logp(model.policy, data)
    mask = data['labels'][:, 1:] != -100
    if algorithm == 'dpo':
        ref = plain_logp(model.reference, data).detach()
        score = (logp-ref).sum(-1)
        expected = -F.logsigmoid(model.beta*(score[:2]-score[2:])).mean()
    else:
        # Exercise clipping, rather than only the ratio-one branch.
        data['old_log_probs'] = data['old_log_probs']+torch.linspace(-0.7, 0.7, logp.numel()).reshape(logp.shape)
        ratio = torch.exp(logp-data['old_log_probs'])
        advantage = data['advantages'] if algorithm == 'ppo' else data['advantages'][:, None]
        policy_loss = -torch.minimum(ratio*advantage, torch.clamp(ratio, .8, 1.2)*advantage)
        if algorithm == 'ppo':
            v = model.critic(data['input_ids'], data['attention_mask'])[:, :-1]
            clipped = data['old_values']+torch.clamp(v-data['old_values'], -.2, .2)
            value_loss = torch.maximum((v-data['returns'])**2, (clipped-data['returns'])**2)
            expected = (policy_loss+0.25*value_loss)[mask].mean()
        else:
            difference = data['reference_log_probs']-logp
            loss = policy_loss+model.kl_coefficient*(torch.exp(difference)-difference-1)
            expected = torch.stack([loss[i][mask[i]].mean() for i in range(mask.shape[0])]).mean()
    actual = model(**data)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual, [p for p in model.parameters() if p.requires_grad], retain_graph=True, allow_unused=True)
    expected_grads = torch.autograd.grad(expected, [p for p in model.parameters() if p.requires_grad], allow_unused=True)
    for a, b in zip(actual_grads, expected_grads):
        if a is None or b is None:
            assert a is None and b is None
        else:
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('algorithm', ['dpo', 'ppo', 'grpo'])
def test_updates_unequal_microbatches_and_exact_checkpoint_continuation(algorithm, tmp_path):
    model, trainer = make(algorithm)
    data = rollout(algorithm, trainer)
    oracle = copy.deepcopy(model)
    oracle_optimizer = torch.optim.AdamW([p for p in oracle.parameters() if p.requires_grad], lr=.01)
    oracle.train(); oracle(**data).backward(); oracle_optimizer.step(); oracle_optimizer.zero_grad(set_to_none=True)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    indices = ([0, 2], [1, 3]) if algorithm == 'dpo' else ([0], [1, 2, 3])
    micros = [{k: v[list(rows)].clone() for k, v in data.items()} for rows in indices]
    metrics = trainer.train_step(micros)
    assert math.isfinite(metrics['loss']) and metrics['normalization_units'] > 0
    for (name, p), (_, q) in zip(model.named_parameters(), oracle.named_parameters()):
        torch.testing.assert_close(p, q)
        if name.startswith('reference.'):
            torch.testing.assert_close(p, before[name], atol=0, rtol=0)
    assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n.startswith('policy.'))
    if algorithm == 'ppo':
        assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n.startswith('critic.'))
    trainer.data_state = {'epoch': 1, 'pending': data, 'source_cursor': 4}
    saved = trainer.save(tmp_path)
    next_random = torch.rand(3)
    trainer.train_step(micros)
    expected_parameters = {n: p.detach().clone() for n, p in model.named_parameters()}
    expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    restored_model, restored = make(algorithm)
    assert restored.resume(saved) == 1
    torch.testing.assert_close(torch.rand(3), next_random, atol=0, rtol=0)
    assert restored.data_state['source_cursor'] == 4
    for k in data:
        torch.testing.assert_close(restored.data_state['pending'][k], data[k], atol=0, rtol=0)
    restored.train_step(micros)
    for n, p in restored_model.named_parameters():
        torch.testing.assert_close(p, expected_parameters[n], atol=0, rtol=0)
    for key, entry in expected_optimizer['state'].items():
        for name, value in entry.items():
            torch.testing.assert_close(restored.optimizer.state_dict()['state'][key][name], value, atol=0, rtol=0)


def test_gae_terminal_vs_truncated_and_prompt_padding_mask():
    mask = torch.tensor([[False, True, True, False], [False, True, True, False]])
    values = torch.tensor([[100., 1., 2., 4., 900.], [100., 1., 2., 4., 900.]])
    rewards = torch.tensor([[0., 0., 3., 0.], [0., 0., 3., 0.]])
    adv, returns = pt.generalized_advantage_estimation(rewards, values, mask, torch.tensor([True, False]), gamma=1., gae_lambda=1.)
    torch.testing.assert_close(adv, torch.tensor([[0., 2., 1., 0.], [0., 6., 5., 0.]]))
    torch.testing.assert_close(returns, torch.tensor([[0., 3., 3., 0.], [0., 7., 7., 0.]]))


def test_group_zero_variance_and_prompt_validation():
    _, trainer = make('grpo')
    data = batch()
    prepared = trainer.prepare_groups(data, torch.tensor([1., 0., 2., 2.]), torch.tensor([0, 0, 1, 1]))
    torch.testing.assert_close(prepared['advantages'][2:], torch.zeros(2))
    assert prepared['advantages'][0] > 0 and prepared['advantages'][1] < 0
    with pytest.raises(ValueError, match='same prompt'):
        trainer.prepare_groups(data, torch.ones(4), torch.tensor([0, 1, 0, 1]))


@pytest.mark.parametrize('algorithm', ['ppo', 'grpo'])
def test_sampling_and_snapshot_preserve_old_policy(algorithm):
    _, trainer = make(algorithm)
    sampled, terminal = trainer.sample([1, 2], max_new_tokens=3, eos_token_ids=[])
    assert not terminal and sampled['input_ids'].shape == (1, 5)
    pt._response_batch(sampled)
    data = rollout(algorithm, trainer)
    old = data['old_log_probs'].clone()
    trainer.train_step([data]); trainer.train_step([data])
    torch.testing.assert_close(data['old_log_probs'], old, atol=0, rtol=0)
