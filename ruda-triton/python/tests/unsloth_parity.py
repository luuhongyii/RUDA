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
if MODE == "ruda":
    os.environ["TRITON_DEFAULT_BACKEND"] = "ruda"
    os.environ["TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"] = "0"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"

import torch  # noqa: E402

DEV = "cuda"


def seeded(seed):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    return g


def case_rms(out):
    from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm
    g = seeded(1)
    X = torch.randn(512, 576, device=DEV, dtype=torch.bfloat16, generator=g).requires_grad_(True)
    W = (1 + 0.1 * torch.randn(576, device=DEV, dtype=torch.bfloat16, generator=g))
    dY = torch.randn(512, 576, device=DEV, dtype=torch.bfloat16, generator=g)
    Y = Fast_RMS_Layernorm.apply(X, W, 1e-5, False)
    Y.backward(dY.clone())
    out["rms.Y"] = Y.detach()
    out["rms.dX"] = X.grad.detach()


def case_swiglu(out):
    from unsloth.kernels.swiglu import swiglu_DWf_DW_dfg_kernel, swiglu_fg_kernel
    g = seeded(2)
    e = torch.randn(2, 256, 1536, device=DEV, dtype=torch.bfloat16, generator=g)
    gg = torch.randn(2, 256, 1536, device=DEV, dtype=torch.bfloat16, generator=g)
    out["swiglu.h"] = swiglu_fg_kernel(e, gg)
    DW = torch.randn(512, 1536, device=DEV, dtype=torch.bfloat16, generator=g)
    e2, g2 = e.view(512, 1536).clone(), gg.view(512, 1536).clone()
    res = swiglu_DWf_DW_dfg_kernel(DW, e2, g2)
    for i, t in enumerate(res if isinstance(res, (tuple, list)) else (res, )):
        out[f"swiglu.dfg{i}"] = t.clone()


def case_rope(out):
    from unsloth.kernels.rope_embedding import fast_rope_embedding
    g = seeded(3)
    Q = torch.randn(2, 9, 256, 64, device=DEV, dtype=torch.bfloat16, generator=g).requires_grad_(True)
    K = torch.randn(2, 3, 256, 64, device=DEV, dtype=torch.bfloat16, generator=g).requires_grad_(True)
    pos = torch.arange(256, device=DEV, dtype=torch.float32)[:, None]
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, device=DEV, dtype=torch.float32) / 64))
    ang = torch.cat([pos * inv, pos * inv], dim=1)
    cos, sin = ang.cos().to(torch.bfloat16), ang.sin().to(torch.bfloat16)
    Qo, Ko = fast_rope_embedding(Q, K, cos, sin)
    (Qo.float().square().sum() + Ko.float().sum()).backward()
    out["rope.Q"], out["rope.K"] = Qo.detach(), Ko.detach()
    out["rope.dQ"], out["rope.dK"] = Q.grad.detach(), K.grad.detach()


def case_nf4(out):
    from unsloth.kernels.nf4 import dequantize_nf4
    g = seeded(4)
    shape = (1536, 576)
    n = shape[0] * shape[1]
    W = torch.randint(0, 256, ((n + 1) // 2, ), device=DEV, dtype=torch.uint8, generator=g)
    absmax = torch.randint(0, 256, (n // 64, ), device=DEV, dtype=torch.uint8, generator=g)
    code2 = torch.randn(256, device=DEV, generator=g)
    absmax2 = torch.rand(n // 64 // 256 + 1, device=DEV, generator=g)
    offset = torch.tensor([0.0123], device=DEV)
    code = torch.sort(torch.randn(16, device=DEV, generator=g)).values
    out["nf4.bf16"] = dequantize_nf4(W, absmax, code2, absmax2, offset, code, 64, 256, shape, torch.bfloat16)


def case_ce(out):
    """Inductor-generated cross-entropy kernels (the loss path of the LoRA fine-tune)."""
    import torch.nn.functional as F
    g = seeded(5)
    logits = (3 * torch.randn(511, 49152, device=DEV, generator=g)).to(torch.bfloat16).requires_grad_(True)
    labels = torch.randint(0, 49152, (511, ), device=DEV, generator=g)
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
    for case in CASES:
        start = time.time()
        try:
            case(results)
            torch.cuda.synchronize()
            status[case.__name__] = "ok"
        except Exception as error:  # report every case, do not stop at the first failure
            status[case.__name__] = f"{type(error).__name__}: {str(error).splitlines()[0][:300]}"
        print(f"[{mode}] {case.__name__}: {status[case.__name__]} ({time.time() - start:.1f}s)", flush=True)
    torch.save({"results": {k: v.cpu() for k, v in results.items()}, "status": status, "backend": backend},
               os.path.join(outdir, f"{mode}.pt"))


def compare(outdir):
    ref = torch.load(os.path.join(outdir, "official.pt"))
    got = torch.load(os.path.join(outdir, "ruda.pt"))
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
        compare(sys.argv[2])
