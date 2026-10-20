import copy
from datetime import timedelta
import importlib
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from architecture_test_utils import NAME

ep = importlib.import_module(NAME + '.expert_parallel')
checkpoints = importlib.import_module(NAME + '.distributed_checkpoint')
training = importlib.import_module(NAME + '.expert_training')
replicas = importlib.import_module(NAME + '.distributed_training')
mesh_module = importlib.import_module(NAME + '.model_parallel_mesh')
mesh_training = importlib.import_module(NAME + '.model_parallel_training')


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)

    def forward(self, value):
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


def make_experts():
    torch.manual_seed(791)
    experts = nn.ModuleList([Expert() for _ in range(5)])
    experts[2].gate_proj.weight.requires_grad_(False)
    return experts


def batch(rank, step):
    generator = torch.Generator().manual_seed(200 + rank + step * 7)
    value = torch.randn(3, 4, generator=generator, requires_grad=True)
    chosen = [0, 2, 2] if step == 0 else [1, 4, 4]
    if rank == 1:
        chosen = [2, 0, 2] if step == 0 else [4, 1, 4]
    ids = torch.tensor(chosen)[:, None]
    weights = torch.tensor([[.5], [.75], [1.]], requires_grad=True)
    return value, ids, weights


def reference(experts, value, ids, weights):
    output = value * 0
    for identity, expert in enumerate(experts):
        rows = (ids[:, 0] == identity).nonzero().flatten()
        if rows.numel():
            output = output.index_add(0, rows, expert(value.index_select(0, rows)) * weights.index_select(0, rows))
    return output


def close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-7)


def worker(rank, uri, directory, tensor_world):
    torch.set_num_threads(1)
    world = 2 * tensor_world
    dist.init_process_group('gloo', init_method=uri, rank=rank, world_size=world, timeout=timedelta(seconds=90))
    mesh = None
    try:
        if tensor_world == 1:
            group = replicas.ReplicaGroup(device_type='cpu')
        else:
            mesh = mesh_module.ModelParallelMesh(expert_parallel=2, tensor_parallel=tensor_world,
                device_type='cpu', transport='nccl', timeout_seconds=90)
            group = mesh.expert
        originals = make_experts()
        golden = copy.deepcopy(originals)
        bank = ep.ExpertParallelMLP.from_linears(originals, group, group_key='test-separate-experts')
        for expert_id, _, name, layer in bank.projection_items():
            assert layer.weight is getattr(originals[expert_id], name).weight
        if mesh is not None:
            bank.shard_tensor_parallel(mesh.tensor)
        coordinator = (training.ExpertParallelGroup(bank, group) if mesh is None else
                       mesh_training.ModelParallelTrainingGroup(bank, mesh))
        optimizer = torch.optim.AdamW(bank.parameters(), lr=.01, weight_decay=.1, foreach=False)
        golden_optimizer = torch.optim.AdamW(golden.parameters(), lr=.01, weight_decay=.1, foreach=False)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            golden_optimizer.zero_grad(set_to_none=True)
            inputs = [batch(peer, step) for peer in range(2)]
            results = [reference(golden, *values) for values in inputs]
            sum(result.square().sum() for result in results).backward()
            value, ids, weights = batch(group.rank, step)
            actual = bank(value, ids, weights)
            close(actual, results[group.rank])
            actual.square().sum().backward()
            close(value.grad, inputs[group.rank][0].grad)
            close(weights.grad, inputs[group.rank][2].grad)
            coordinator.synchronize_gradients(local_weight=1, normalized=True)
            for expert_id, column, name, layer in bank.projection_items():
                expected = getattr(golden[expert_id], name).weight.grad
                if expected is None:
                    assert layer.weight.grad is None
                else:
                    if mesh is not None:
                        axis = 0 if column < 2 else 1
                        expected = expected.chunk(tensor_world, dim=axis)[mesh.tensor.rank]
                    close(layer.weight.grad, expected)
            coordinator.prepare_optimizer_step(optimizer)
            optimizer.step()
            golden_optimizer.step()
            for expert_id, column, name, layer in bank.projection_items():
                expected_parameter = getattr(golden[expert_id], name).weight
                expected = expected_parameter.detach()
                if mesh is not None:
                    expected = expected.chunk(tensor_world, dim=0 if column < 2 else 1)[mesh.tensor.rank]
                close(layer.weight, expected)
                assert (layer.weight in optimizer.state) == (expected_parameter in golden_optimizer.state)
                if layer.weight in optimizer.state:
                    close(optimizer.state[layer.weight]['step'], golden_optimizer.state[expected_parameter]['step'])

        checkpoint = checkpoints.DistributedCheckpoint(coordinator, directory)
        checkpoint.save(bank, optimizer, step=2, application_state={'rank': rank})
        saved = copy.deepcopy(bank.state_dict())
        with torch.no_grad():
            for parameter in bank.parameters():
                parameter.add_(3)
        step, application = checkpoint.load(bank, optimizer)
        assert step == 2 and application == {'rank': rank}
        for name, value in bank.state_dict().items():
            close(value, saved[name])
        full = checkpoint.consolidate()
        for name, expected in golden.state_dict().items():
            close(full['model'][name], expected)
        # Restore into a different expert topology using actual singleton groups.
        singletons = [dist.new_group([peer], backend='gloo') for peer in range(world)]
        singleton = replicas.ReplicaGroup(singletons[rank], device_type='cpu')
        restored = ep.ExpertParallelMLP.from_linears(make_experts(), singleton,
            group_key='test-restored-experts')
        resumed = torch.optim.AdamW(restored.parameters(), lr=.2, foreach=False)
        local = checkpoints.DistributedCheckpoint(training.ExpertParallelGroup(restored, singleton), directory)
        local.load_consolidated(restored, resumed, full, reshard_application=lambda states, rank, world: {})
        for name, value in restored.state_dict().items():
            close(value, golden.state_dict()[name])
        for name, parameter in restored.named_parameters():
            expected = golden.get_parameter(name)
            assert (parameter in resumed.state) == (expected in golden_optimizer.state)
            if parameter in resumed.state:
                for field, value in resumed.state[parameter].items():
                    close(value, golden_optimizer.state[expected][field])
    finally:
        if mesh is not None:
            mesh.close()
        dist.destroy_process_group()


@pytest.mark.parametrize('tensor_world', [1, 2])
def test_separate_experts_training_and_resharding(tmp_path, tensor_world):
    mp.spawn(worker, args=((tmp_path / 'rendezvous').resolve().as_uri(), str(tmp_path / 'checkpoint'), tensor_world),
             nprocs=2 * tensor_world, join=True)


def test_ownership_rejects_missing_or_wrong_rank():
    ownership = importlib.import_module(NAME + '._expert_ownership')
    spec = dict(name='2.weight', partition='expert-owned', expert_id=2, num_experts=5)
    assert ownership.merge_layouts([{}, {'2.weight': spec}]) == {'2.weight': spec}
    with pytest.raises(ValueError, match='ownership'):
        ownership.merge_layouts([{'2.weight': spec}, {}])
