"""Single-A100 real-checkpoint/real-text LoRA training comparison worker."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback
import types

import torch


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def modules(backend):
    if backend == 'ruda':
        import ruda_torch as r
        return r, importlib.import_module('ruda_torch.finetuning'), importlib.import_module('ruda_torch.causal_finetuning')
    # Execute the exact shared production Python modules on real torch.cuda
    # tensors without requiring the separately built RUDA native extension.
    name = 'ruda_cuda_training_reference'
    package = types.ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / 'python/ruda_torch')]
    package._graph_available = package._training_available = package._training_fusion_available = False
    sys.modules[name] = package
    return None, importlib.import_module(name + '.finetuning'), importlib.import_module(name + '.causal_finetuning')


class MemorySamples:
    def __init__(self):
        import pynvml
        pynvml.nvmlInit()
        self.nvml = pynvml
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.stop = threading.Event()
        self.peak = 0
        self.samples = 0
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            used = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
            self.peak = max(self.peak, used); self.samples += 1
            self.stop.wait(.02)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set(); self.thread.join()
        return dict(nvml_sampled_device_peak_bytes=self.peak, nvml_samples=self.samples,
                    nvml_sample_interval_seconds=.02,
                    memory_scope='whole single GPU, including driver and allocator reservations; sampled, not exact peak')


def main(args):
    from transformers import AutoModelForCausalLM
    args.output.mkdir(parents=True, exist_ok=True)
    state_file = args.output / 'benchmark_status.json'
    def status(stage, **details):
        value = dict(stage=stage, backend=args.backend, **details)
        write(state_file, value); print(json.dumps(value), flush=True)
    assert torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    inputs = torch.load(args.inputs, map_location='cpu', weights_only=True)
    tokens = inputs['input_ids']
    assert tuple(tokens.shape) == (args.warmup + args.steps, 1, 512)
    model_dir = inputs['provenance']['model_path']
    r, ft, cft = modules(args.backend)
    device = 'ruda:0' if r else 'cuda:0'
    status('load_model', revision=revision, model=model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16,
        attn_implementation='eager' if r else 'sdpa', low_cpu_mem_usage=True, local_files_only=True)
    assert model.config.model_type in ('qwen2', 'qwen3', 'llama', 'mistral')
    assert model.config.attention_dropout == 0
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    torch.manual_seed(20261024)
    ft.inject_lora(model, target_modules='all-linear', rank=16, alpha=16., adapter_dtype=torch.float32)
    initial = hashlib.sha256()
    trainable_parameters = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.dtype == torch.float32
            initial.update(name.encode()); initial.update(parameter.detach().numpy().tobytes())
            trainable_parameters += parameter.numel()
    model.to(device).train()
    backbone, head = model.get_decoder(), model.get_output_embeddings()
    checkpoint_paths = [f'layers.{i}' for i in range(len(backbone.layers))]
    options = dict(token_chunk_size=32, activation_checkpointing=True,
                   checkpoint_modules=checkpoint_paths, preserve_rng_state=True)
    if r:
        assert r._training_fusion_available, 'matching native fusion API 3 required'
        loss_model = r.prepare_causal_lm(model, ruda_attention='native', **options)
        optimizer = r.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4,
            betas=(.9, .999), eps=1e-8, weight_decay=.01, fused_step=True, hierarchical_stats=True)
        synchronize = r.synchronize
    else:
        loss_model = cft.CausalLMFinetuner(backbone, head, **options)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4,
            betas=(.9, .999), eps=1e-8, weight_decay=.01, fused=True)
        synchronize = torch.cuda.synchronize
    executable = loss_model
    if args.backend == 'cuda_compile':
        status('prepare_inductor')
        executable = torch.compile(loss_model, backend='inductor', fullgraph=True, dynamic=False)
    batches = [block.to(device) for block in tokens]
    from torch.nn.attention import SDPBackend, sdpa_kernel
    attention_context = nullcontext() if r else sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    samples = []; losses = []; warmup_seconds = []
    sampler = MemorySamples(); sampler.start()
    try:
        with attention_context:
            for index, ids in enumerate(batches):
                optimizer.zero_grad(set_to_none=True)
                synchronize()
                status('warmup' if index < args.warmup else 'measuring', step=index)
                started = time.perf_counter()
                loss = executable(ids, None, ids)
                loss.backward()
                if index == 0:
                    synchronize()
                    snapshot = {name: p.grad.detach().cpu() for name, p in model.named_parameters()
                                if p.requires_grad and p.grad is not None}
                    assert len(snapshot) == sum(p.requires_grad for p in model.parameters())
                    assert snapshot and all(torch.isfinite(value).all() for value in snapshot.values())
                    torch.save(snapshot, args.output / (args.backend + '_first_gradients.pt'))
                    status('first_step_gradients_saved', loss=float(loss.detach().cpu()))
                optimizer.step()
                synchronize()
                elapsed = time.perf_counter() - started
                value = float(loss.detach().cpu())
                assert value == value and abs(value) != float('inf')
                losses.append(value)
                if index < args.warmup:
                    warmup_seconds.append(elapsed)
                else:
                    samples.append(elapsed)
                status('step_complete', step=index, measured=index >= args.warmup,
                       seconds=elapsed, loss=value, input_tokens_per_second=512 / elapsed)
    finally:
        memory = sampler.finish()
    med = statistics.median(samples)
    result = dict(status='passed', backend=args.backend, source_revision=revision,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
        versions={name: version(name) for name in ('transformers', 'accelerate', 'safetensors')},
        seed=20261024, base_dtype='bfloat16', adapter_dtype='float32', lora_rank=16, lora_alpha=16.,
        base_parameters=base_parameters,
        model_type=model.config.model_type, model_config=model.config.to_dict(),
        trainable_tensors=sum(p.requires_grad for p in model.parameters()),
        lora_targets='all-linear including output head', trainable_parameters=trainable_parameters,
        initial_adapter_sha256=initial.hexdigest(), data=inputs['provenance'],
        batch_size=1, sequence_length=512, supervised_tokens_per_step=511, warmup_steps=args.warmup,
        measured_steps=args.steps, step_seconds=samples, warmup_seconds=warmup_seconds, losses=losses,
        median_step_seconds=med, median_input_tokens_per_second=512 / med,
        median_supervised_tokens_per_second=511 / med,
        timing_scope='forward + backward + optimizer update + final synchronization; excludes data transfer and zero_grad',
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        attention='RUDA fused online dense' if r else 'PyTorch Flash SDPA',
        optimizer='RUDA fused_step hierarchical_stats' if r else 'PyTorch AdamW fused=True',
        activation_checkpointing='same non-reentrant decoder-layer wrappers',
        tf32=False, compiled=args.backend == 'cuda_compile', **memory)
    if r:
        result['ruda_execution_stats'] = r.execution_stats()
        result['native_library_sha256'] = hashlib.sha256(Path(r._library).read_bytes()).hexdigest()
        result['cpp_extension_sha256'] = hashlib.sha256(Path(r._C.__file__).read_bytes()).hexdigest()
        result['training_fusion_api_version'] = r._C.training_fusion_api_version
    write(args.output / (args.backend + '.json'), result)
    status('passed', result_file=str(args.output / (args.backend + '.json')),
           median_step_seconds=med, input_tokens_per_second=512 / med)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('ruda', 'cuda', 'cuda_compile'), required=True)
    parser.add_argument('--inputs', type=Path, default=Path('/content/ruda_validation/training_inputs.pt'))
    parser.add_argument('--output', type=Path, default=Path('/content/ruda_validation'))
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--steps', type=int, default=5)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write(args.output / (args.backend + '.json'), dict(status='failed', backend=args.backend,
              error_type=type(error).__name__, error=str(error)))
        write(args.output / 'benchmark_status.json', dict(stage='failed', backend=args.backend,
              error_type=type(error).__name__, error=str(error)))
        traceback.print_exc()
        sys.exit(1)
