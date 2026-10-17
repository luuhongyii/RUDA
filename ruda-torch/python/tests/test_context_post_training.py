"""Real Gloo CP objectives against full SDPA, not a native ring GPU test."""
import importlib
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as collective
import torch.multiprocessing as multiprocessing
from torch import nn
from torch.nn import functional as F
from architecture_test_utils import NAME

pt = importlib.import_module(NAME+'.post_training')
cft = importlib.import_module(NAME+'.causal_finetuning')
data_ops = importlib.import_module(NAME+'.training_data')
mesh_ops = importlib.import_module(NAME+'.model_parallel_mesh')
training_ops = importlib.import_module(NAME+'.model_parallel_training')
torch.set_num_threads(1)


class SDPAReference(nn.Module):
    def __init__(self, group=None):
        super().__init__()
        self.embedding = nn.Embedding(13, 6)
        self.positions = nn.Embedding(32, 6)
        self.qkv = nn.Linear(6, 18, bias=False)
        self.output = nn.Linear(6, 6, bias=False)
        self.group = group
        self.length_scale = 0.

    def forward(self, input_ids, attention_mask, position_ids=None, cu_seqlens=None,
                max_seqlen=None, ruda_context_positions=None, ruda_context_documents=None,
                ruda_context_valid=None):
        packed = input_ids.ndim == 1
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[-1]).expand_as(input_ids)
        if packed:
            documents = torch.repeat_interleave(torch.arange(cu_seqlens.numel()-1), cu_seqlens.diff())
        else:
            documents = ruda_context_documents if self.group else torch.arange(input_ids.shape[0])[:, None].expand_as(input_ids)
        ids = input_ids[None] if packed else input_ids
        positions = position_ids[None] if packed else position_ids
        documents = documents[None] if packed else documents
        valid = attention_mask[None] if packed else attention_mask
        x = self.embedding(ids)+self.positions(positions)
        q, k, v = self.qkv(x).chunk(3, -1)
        keys, values, key_positions, key_documents, key_valid = k, v, positions, documents, valid
        if self.group:
            sizes = self.group.gather_metadata(ids.shape[1])
            maximum = max(sizes)
            payload = torch.stack((F.pad(k, (0, 0, 0, maximum-k.shape[1])),
                                   F.pad(v, (0, 0, 0, maximum-v.shape[1]))), 0)
            peers = collective.all_gather(payload, group=self.group.group)
            payload = torch.cat([peer[:, :, :length] for peer, length in zip(peers, sizes)], 2)
            keys, values = payload.unbind(0)
            def gather_metadata(value):
                padded = F.pad(value, (0, maximum-value.shape[1])).contiguous()
                peers = [torch.empty_like(padded) for _ in sizes]
                dist.all_gather(peers, padded, group=self.group.group)
                return torch.cat([peer[:, :length] for peer, length in zip(peers, sizes)], 1)
            key_positions = gather_metadata(positions)
            key_documents = gather_metadata(documents)
            key_valid = gather_metadata(valid)
        mask = (positions[:, :, None] >= key_positions[:, None, :]) & (documents[:, :, None] == key_documents[:, None, :])
        mask = mask & key_valid[:, None, :]
        attended = F.scaled_dot_product_attention(q[:, None], keys[:, None], values[:, None],
            attn_mask=mask[:, None]).squeeze(1)
        length = int(key_positions.max())+1 if key_positions.numel() else 0
        result = (x+self.output(attended))*valid[..., None]*(1+self.length_scale*length)
        return result.squeeze(0) if packed else result


class CPPolicy(cft.ContextParallelFinetuner):
    def __init__(self, backbone, head, group):
        # This explicit reference already implements CP SDPA. Production HF
        # adapters/native ring are not replaced or exercised by this CPU test.
        cft.CausalLMFinetuner.__init__(self, backbone, head, token_chunk_size=2)
        self.context_group, self.context_key = group, 'cpu-post-training-reference'


class CPValue(pt.ContextParallelValueModel):
    def __init__(self, backbone, head, group):
        cft.CausalLMFinetuner.__init__(self, backbone, head, token_chunk_size=2)
        self.context_group, self.context_key = group, 'cpu-post-training-value-reference'


def make(algorithm, packed, prefix, mesh=None):
    torch.manual_seed(781)
    group = mesh.context if mesh else None
    def policy(value=False):
        backbone = SDPAReference(group)
        backbone.length_scale = .03 if prefix else 0.
        head = nn.Linear(6, 1 if value else 13, bias=False)
        if group:
            return (CPValue if value else CPPolicy)(backbone, head, group)
        cls = (pt.PackedCausalValueModel if packed else pt.CausalValueModel) if value else (
            cft.PackedCausalLMFinetuner if packed else cft.CausalLMFinetuner)
        return cls(backbone, head, token_chunk_size=2)
    if algorithm == 'reward':
        model, trainer_cls = pt.RewardModel(policy(True), prefix_scoring=prefix), pt.RewardTrainer
    else:
        actor, reference = policy(), policy()
        reference.load_state_dict(actor.state_dict())
        with torch.no_grad():
            reference.head.weight.add_(torch.randn_like(reference.head.weight)*.1)
        if algorithm == 'dpo':
            model, trainer_cls = pt.DPOModel(actor, reference, reference_id='ref', beta=.4, prefix_scoring=prefix), pt.DPOTrainer
        elif algorithm == 'ppo':
            model = pt.PPOModel(actor, reference, policy(True), reference_id='ref', kl_coefficient=.1,
                                gamma=.9, gae_lambda=.7, prefix_scoring=prefix)
            trainer_cls = pt.PPOTrainer
        else:
            model, trainer_cls = pt.GRPOModel(actor, reference, reference_id='ref', prefix_scoring=prefix), pt.GRPOTrainer
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01, momentum=.9)
    coordinator = training_ops.ModelParallelTrainingGroup(model, mesh) if mesh else None
    trainer = trainer_cls(model, optimizer, base_id='base', run_config={'source': 'exact'}, replica_group=coordinator)
    return model, trainer


def response_batch(packed):
    ids = [[1, 2, 3, 4, 5], [1, 2, 6], [3, 4, 7, 8], [3, 4, 9, 10, 11, 12, 1]]
    samples = [dict(input_ids=row, labels=[-100, -100]+row[2:]) for row in ids]
    return data_ops.collate_varlen_causal_lm(samples) if packed else cft.SFTCollator(max_length=12, pad_token_id=0)(samples)


def prepare(algorithm, trainer, packed):
    batch = response_batch(packed)
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


def equal_batch(actual, expected):
    assert set(actual) == set(expected)
    for name in actual:
        if isinstance(actual[name], torch.Tensor):
            torch.testing.assert_close(actual[name], expected[name], atol=3e-6, rtol=3e-5, msg=name)
        else:
            assert actual[name] == expected[name]


def worker(rank, uri, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=uri, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    mesh = None
    try:
        mesh = mesh_ops.ModelParallelMesh(context_parallel=2, device_type='cpu', transport='nccl', timeout_seconds=120)
        for algorithm in ('reward', 'dpo', 'ppo', 'grpo'):
            for packed in (False, True):
                for prefix in (False, True):
                    model, trainer = make(algorithm, packed, prefix, mesh)
                    oracle, dense = make(algorithm, packed, prefix)
                    oracle.load_state_dict(model.state_dict())
                    fixed = prepare(algorithm, trainer, packed)
                    expected = prepare(algorithm, dense, packed)
                    source = fixed['response_metadata']['source'] if 'response_metadata' in fixed else fixed
                    equal_batch(source, expected)
                    if algorithm == 'reward':
                        torch.testing.assert_close(trainer.score(fixed), dense.score(expected), atol=3e-6, rtol=3e-5)
                        torch.testing.assert_close(trainer.token_rewards(fixed), dense.token_rewards(expected), atol=3e-6, rtol=3e-5)
                    splits = ([0, 2], [1, 3]) if algorithm in ('dpo', 'reward') else ([0], [1, 2, 3])
                    local = [pt.select_response_batch(fixed, indices) for indices in splits]
                    result = trainer.train_step(local)
                    reference = dense.train_step([expected])
                    assert result['normalization_units'] == reference['normalization_units']
                    torch.testing.assert_close(torch.tensor(result['loss']), torch.tensor(reference['loss']), atol=3e-6, rtol=3e-5,
                        msg=f'{algorithm} packed={packed} prefix={prefix}')
                    for (name, actual), (_, wanted) in zip(model.named_parameters(), oracle.named_parameters()):
                        torch.testing.assert_close(actual, wanted, atol=3e-6, rtol=3e-5, msg=name)
                        if actual.grad is not None:
                            torch.testing.assert_close(actual.grad, wanted.grad, atol=3e-6, rtol=3e-5, msg=name+' gradient')
                    trainer.data_state = {'fixed': local, 'epoch': 1}
                    root = Path(directory)/f'{algorithm}-{packed}-{prefix}'
                    trainer.save_distributed(root)
                    trainer.train_step(local)
                    continued = [p.detach().clone() for p in model.parameters()]
                    trainer.resume_distributed(root)
                    assert trainer.step == 1 and trainer.data_state['epoch'] == 1
                    trainer.train_step(trainer.data_state['fixed'])
                    for actual, wanted in zip(model.parameters(), continued):
                        torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
                    if algorithm in ('ppo', 'grpo'):
                        oracle.load_state_dict(model.state_dict())
                        sample, terminal = trainer.sample([1], max_new_tokens=2, eos_token_ids=[], generator=torch.Generator().manual_seed(83))
                        plain, plain_terminal = dense.sample([1], max_new_tokens=2, eos_token_ids=[], generator=torch.Generator().manual_seed(83))
                        assert sample['input_ids'].reshape(-1).tolist() == plain['input_ids'].reshape(-1).tolist()
                        assert terminal == plain_terminal == False
                        eos = int(sample['input_ids'].reshape(-1)[1])
                        sample, terminal = trainer.sample([1], max_new_tokens=2, eos_token_ids=[eos], generator=torch.Generator().manual_seed(83))
                        assert terminal and sample['input_ids'].numel() == 2
    finally:
        if mesh is not None:
            mesh.close()
        dist.destroy_process_group()


def test_context_objectives_snapshots_gradients_microbatches_sampling_and_resume(tmp_path):
    multiprocessing.spawn(worker, args=((tmp_path/'rendezvous').resolve().as_uri(), str(tmp_path/'checkpoint')),
                          nprocs=2, join=True)
