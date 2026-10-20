"""Kernel-by-kernel parity of Unsloth's Triton kernels: official Triton vs the Ruda backend.

    python unsloth_parity.py run official OUT
    python unsloth_parity.py run ruda OUT
    python unsloth_parity.py compare OUT

Each ``run`` executes the same seeded inputs in its own process (one Triton driver per process)
and saves every output; ``compare`` reports max error and the fraction of bit-identical values.
"""
import os
import sys
import time

MODE = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "run" else None
if MODE in ("ruda", "gx1"):
    os.environ["TRITON_DEFAULT_BACKEND"] = "ruda"
    os.environ["TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"] = "0"
if MODE == "gx1":
    os.environ["RUDA_TRITON_DEVICE"] = "gx1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"

import contextlib  # noqa: E402

import torch  # noqa: E402

# Inputs are always generated on CUDA so every mode sees the same values; ``gx1`` runs the
# kernels on the GX1 functional model with CPU tensors.
GEN = "cuda"
DEV = "cpu" if MODE == "gx1" else "cuda"
if MODE == "gx1":
    import torch._inductor.config as inductor_config
    inductor_config.cpu_backend = "triton"
    import unsloth.kernels.utils  # noqa: F401
    for name in ("rms_layernorm", "swiglu", "rope_embedding", "utils"):
        module = sys.modules.get(f"unsloth.kernels.{name}") or __import__(f"unsloth.kernels.{name}", fromlist=["x"])
        module.torch_gpu_device = lambda device: contextlib.nullcontext()


def seeded(seed):
    g = torch.Generator(device=GEN)
    g.manual_seed(seed)
    return g


def case_rms(out):
    from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm
    g = seeded(1)
    X = torch.randn(512, 576, device=GEN, dtype=torch.bfloat16, generator=g).to(DEV).requires_grad_(True)
    W = (1 + 0.1 * torch.randn(576, device=GEN, dtype=torch.bfloat16, generator=g)).to(DEV)
    dY = torch.randn(512, 576, device=GEN, dtype=torch.bfloat16, generator=g).to(DEV)
    Y = Fast_RMS_Layernorm.apply(X, W, 1e-5, False)
    Y.backward(dY.clone())
    out["rms.Y"] = Y.detach()
    out["rms.dX"] = X.grad.detach()


def case_swiglu(out):
    from unsloth.kernels.swiglu import swiglu_DWf_DW_dfg_kernel, swiglu_fg_kernel
    g = seeded(2)
    e = torch.randn(2, 256, 1536, device=GEN, dtype=torch.bfloat16, generator=g).to(DEV)
    gg = torch.randn(2, 256, 1536, device=GEN, dtype=torch.bfloat16, generator=g).to(DEV)
    out["swiglu.h"] = swiglu_fg_kernel(e, gg)
    DW = torch.randn(512, 1536, device=GEN, dtype=torch.bfloat16, generator=g).to(DEV)
    e2, g2 = e.view(512, 1536).clone(), gg.view(512, 1536).clone()
    res = swiglu_DWf_DW_dfg_kernel(DW, e2, g2)
    for i, t in enumerate(res if isinstance(res, (tuple, list)) else (res, )):
        out[f"swiglu.dfg{i}"] = t.clone()


def case_rope(out):
    from unsloth.kernels.rope_embedding import fast_rope_embedding
    g = seeded(3)
    Q = torch.randn(2, 9, 256, 64, device=GEN, dtype=torch.bfloat16, generator=g).to(DEV).requires_grad_(True)
    K = torch.randn(2, 3, 256, 64, device=GEN, dtype=torch.bfloat16, generator=g).to(DEV).requires_grad_(True)
    pos = torch.arange(256, device=GEN, dtype=torch.float32)[:, None]
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, device=GEN, dtype=torch.float32) / 64))
    ang = torch.cat([pos * inv, pos * inv], dim=1)
    cos, sin = ang.cos().to(torch.bfloat16).to(DEV), ang.sin().to(torch.bfloat16).to(DEV)
    Qo, Ko = fast_rope_embedding(Q, K, cos, sin)
    (Qo.float().square().sum() + Ko.float().sum()).backward()
    out["rope.Q"], out["rope.K"] = Qo.detach(), Ko.detach()
    out["rope.dQ"], out["rope.dK"] = Q.grad.detach(), K.grad.detach()


def case_nf4(out):
    from unsloth.kernels.nf4 import dequantize_nf4
    g = seeded(4)
    shape = (1536, 576)
    n = shape[0] * shape[1]
    W = torch.randint(0, 256, ((n + 1) // 2, ), device=GEN, dtype=torch.uint8, generator=g).to(DEV)
    absmax = torch.randint(0, 256, (n // 64, ), device=GEN, dtype=torch.uint8, generator=g).to(DEV)
    code2 = torch.randn(256, device=GEN, generator=g).to(DEV)
    absmax2 = torch.rand(n // 64 // 256 + 1, device=GEN, generator=g).to(DEV)
    offset = torch.tensor([0.0123], device=DEV)
    code = torch.sort(torch.randn(16, device=GEN, generator=g).to(DEV)).values
    if MODE == "gx1":
        # dequantize_nf4 guards the launch with torch.cuda.device; call its launcher directly.
        from unsloth.kernels import nf4
        buffer = torch.empty(shape, dtype=torch.bfloat16, device=DEV)
        out["nf4.bf16"] = nf4._launch(nf4._nf4_dequant_kernel, W, absmax, code2, absmax2, offset, code, 64, 256,
                                      buffer, fp_fusion=False)
        return
    out["nf4.bf16"] = dequantize_nf4(W, absmax, code2, absmax2, offset, code, 64, 256, shape, torch.bfloat16)


def case_ce(out):
    """Inductor-generated cross-entropy kernels (the loss path of the LoRA fine-tune)."""
    import torch.nn.functional as F
    g = seeded(5)
    logits = (3 * torch.randn(511, 49152, device=GEN, generator=g).to(DEV)).to(torch.bfloat16).requires_grad_(True)
    labels = torch.randint(0, 49152, (511, ), device=GEN, generator=g).to(DEV)
    labels[::7] = -100
    loss = torch.compile(lambda x, y: F.cross_entropy(x.float(), y, ignore_index=-100))(logits, labels)
    loss.backward()
    out["ce.loss"] = loss.detach().reshape(1)
    out["ce.dlogits"] = logits.grad.detach()


CASES = [case_rms, case_swiglu, case_rope, case_nf4, case_ce]


def run(mode, outdir):
    os.makedirs(outdir, exist_ok=True)
    import triton
    backend = triton.runtime.driver.active.get_current_target().backend
    print(f"[{mode}] triton driver backend: {backend}", flush=True)
    results, status = {}, {}
    launches = []
    if mode == "gx1":
        from ruda_triton import gx1
        original = gx1.launch

        def counted(handle, *args, **kwargs):
            launches.append(handle)
            return original(handle, *args, **kwargs)

        gx1.launch = counted
    only = os.environ.get("PARITY_CASES")
    for case in CASES:
        if only and case.__name__.removeprefix("case_") not in only.split(","):
            continue
        start = time.time()
        try:
            case(results)
            torch.cuda.synchronize()
            status[case.__name__] = "ok"
        except Exception as error:  # report every case, do not stop at the first failure
            import traceback
            traceback.print_exc()
            status[case.__name__] = f"{type(error).__name__}: {(str(error).splitlines() or [''])[0][:300]}"
        extra = f", {len(launches)} GX1 launches of {len(set(launches))} kernels" if mode == "gx1" else ""
        launches.clear()
        print(f"[{mode}] {case.__name__}: {status[case.__name__]} ({time.time() - start:.1f}s{extra})", flush=True)
    torch.save({"results": {k: v.cpu() for k, v in results.items()}, "status": status, "backend": backend},
               os.path.join(outdir, f"{mode}.pt"))


def compare(outdir, mode="ruda"):
    ref = torch.load(os.path.join(outdir, "official.pt"))
    got = torch.load(os.path.join(outdir, f"{mode}.pt"))
    print(f"official backend={ref['backend']}  ruda backend={got['backend']}")
    for name, st in got["status"].items():
        print(f"  {name}: official={ref['status'].get(name)}  ruda={st}")
    print(f"{'output':<16}{'shape':<22}{'dtype':<16}{'max_abs':>12}{'max_rel':>12}{'bit_equal':>11}")
    for key, a in ref["results"].items():
        if key not in got["results"]:
            print(f"{key:<16}missing in ruda run")
            continue
        b = got["results"][key]
        af, bf = a.float(), b.float()
        diff = (af - bf).abs()
        rel = diff / af.abs().clamp_min(1e-6)
        equal = (a.view(torch.uint8 if a.element_size() == 1 else torch.int16 if a.element_size() == 2 else torch.int32)
                 == b.view(torch.uint8 if b.element_size() == 1 else torch.int16 if b.element_size() == 2 else torch.int32))
        print(f"{key:<16}{str(tuple(a.shape)):<22}{str(a.dtype):<16}{diff.max().item():>12.3e}"
              f"{rel.max().item():>12.3e}{equal.float().mean().item():>10.4%}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        compare(*sys.argv[2:4])
