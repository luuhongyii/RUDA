import importlib
from pathlib import Path
import sys
import types

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

name='ruda_compile_patterns_host_test'
package=types.ModuleType(name);package.__path__=[str(Path(__file__).resolve().parents[1]/'ruda_torch')]
sys.modules[name]=package
patterns=importlib.import_module(name+'._compile_patterns')
planner=importlib.import_module(name+'._graph_opt')


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_rms_pattern_preserves_storage_casts_and_public_intermediates(dtype):
    def function(x,gain):
        f=x.float();square=f.square();mean=square.mean(-1,keepdim=True)
        return (f*torch.rsqrt(mean+1e-5)).to(x.dtype)*gain,mean
    args=(torch.randn(3,97).to(dtype),torch.randn(97).to(dtype))
    gm=make_fx(function)(*args);expected=gm(*args)
    gm=patterns.fuse_normalizations(gm)
    assert sum(n.target==torch.ops.aten.rms_norm.default for n in gm.graph.nodes)==1
    for actual,reference in zip(gm(*args),expected,strict=True):
        torch.testing.assert_close(actual,reference,rtol=.02 if dtype!=torch.float32 else 2e-6,atol=.02 if dtype!=torch.float32 else 2e-6)


@pytest.mark.parametrize('axis,eps',[(0,1e-5),(-1,-1e-5),(-1,1e-100)])
def test_other_normalization_semantics_are_not_rewritten(axis,eps):
    def function(x):return x*torch.rsqrt(x.square().mean(axis,keepdim=True)+eps)
    gm=patterns.fuse_normalizations(make_fx(function)(torch.randn(3,97)))
    assert not any(n.target==torch.ops.aten.rms_norm.default for n in gm.graph.nodes)


def test_internal_copy_is_removed_but_public_copy_is_preserved():
    T=planner.TensorSpec;Op=planner.GraphOp
    plan=planner.prepare_plan({'x':T((3,97),'bfloat16'),'up':T((3,97),'bfloat16')},
        [Op.copy('inner','x'),Op.silu('gate','inner'),Op.mul('result','gate','up'),Op.copy('saved','result')],
        outputs=['result','saved'],optimize=True,reuse_workspace=True)
    assert [n.kind for n in plan.nodes]==['silu_mul','copy']
    assert plan.nodes[0].left=='x' and 'inner' in plan.eliminated_outputs
    assert plan.storage_roots[-1]!=plan.storage_roots[-2]
