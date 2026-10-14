"""Route RUDA SDPA to its differentiable native operator before AOTAutograd."""
import torch
from torch.fx import Graph,GraphModule,Node


def _sdpa(query,key,value,attn_mask=None,dropout_p=0.0,is_causal=False,*,scale=None,enable_gqa=False):
    from .attention import scaled_dot_product_attention
    return scaled_dot_product_attention(query,key,value,attn_mask=attn_mask,dropout_p=dropout_p,
        is_causal=is_causal,scale=scale,enable_gqa=enable_gqa)


def route_attention(gm):
    targets={torch.nn.functional.scaled_dot_product_attention,torch.ops.aten.scaled_dot_product_attention.default}
    selected=set()
    for node in gm.graph.nodes:
        if node.op!='call_function' or node.target not in targets:continue
        query=node.args[0] if node.args else node.kwargs.get('query')
        value=query.meta.get('val',query.meta.get('example_value')) if isinstance(query,Node) else None
        if (isinstance(value,torch.Tensor) and value.device.type=='ruda' and value.layout==torch.strided
                and value.dtype in (torch.float32,torch.float16,torch.bfloat16) and value.ndim>=3):selected.add(node)
    if not selected:return gm
    graph=Graph();env={}
    for node in gm.graph.nodes:
        copied=graph.node_copy(node,lambda n:env[n])
        if node in selected:copied.target=_sdpa
        copied.meta=dict(node.meta);env[node]=copied
    return GraphModule(gm,graph)
