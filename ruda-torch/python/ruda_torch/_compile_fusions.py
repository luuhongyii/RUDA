"""Connect model-independent SwiGLU expressions before AOT builds backward."""
import operator

import torch
from torch.fx import Node


def _silu_mul(gate, up):
    from .training import silu_mul
    return silu_mul(gate.contiguous(), up.contiguous())


def _value(node):
    return node.meta.get('val', node.meta.get('example_value')) if isinstance(node, Node) else None


def route_byte_assignments(gm):
    changed=False
    for node in gm.graph.nodes:
        if node.op!='call_function' or node.target is not operator.setitem or len(node.args)!=3 or node.kwargs or node.users:
            continue
        destination,mask,code=node.args
        if isinstance(mask,(tuple,list)) and len(mask)==1:mask=mask[0]
        target,condition=_value(destination),_value(mask)
        if (not isinstance(target,torch.Tensor) or not isinstance(condition,torch.Tensor)
                or target.device.type!='ruda' or condition.device!=target.device
                or target.dtype!=torch.uint8 or condition.dtype!=torch.bool
                or target.layout!=torch.strided or not target.is_contiguous()
                or any(type(d) is not int for d in (*target.shape,*condition.shape))
                or tuple(condition.shape)!=tuple(target.shape)
                or type(code) is not int or not 0<=code<=255):continue
        node.target=torch.ops.aten.masked_fill_.Scalar
        node.args=(destination,mask,code)
        node.meta=dict(destination.meta)
        changed=True
    if changed:gm.recompile()
    return gm


def _operands(node):
    if (node.op == 'call_function' and node.target in
            (operator.mul, torch.mul, torch.ops.aten.mul.Tensor)
            or node.op == 'call_method' and node.target == 'mul'):
        values = dict(zip(('input', 'other'), node.args))
        values.update(node.kwargs)
        if len(node.args) <= 2 and set(values) == {'input', 'other'}:
            return values['input'], values['other']
    return None


def _gate(node):
    if not isinstance(node, Node) or node.op != 'call_function':
        return None
    if node.target == torch.ops.aten.silu.default and len(node.args) == 1 and not node.kwargs:
        return node.args[0]
    if node.target == torch.nn.functional.silu:
        values = dict(zip(('input', 'inplace'), node.args))
        values.update(node.kwargs)
        if (len(node.args) <= 2 and 'input' in values and not set(values) - {'input', 'inplace'}
                and values.get('inplace', False) is False):
            return values['input']
    return None


def _compatible(gate, activation, up, output):
    values = [_value(node) for node in (gate, activation, up, output)]
    first = values[0]
    if (not isinstance(first, torch.Tensor) or first.device.type != 'ruda'
            or first.device.index not in (None, 0)
            or first.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or any(type(d) is not int or d <= 0 for d in first.shape)):
        return False
    return all(isinstance(value, torch.Tensor) and value.layout == torch.strided
               and value.device == first.device and value.dtype == first.dtype
               and tuple(value.shape) == tuple(first.shape)
               and not value.is_conj() and not value.is_neg() for value in values)


def _can_delay(activation, output):
    # Do not move the activation across arbitrary Python calls or mutations.
    node = activation.next
    while node is not output:
        if node.op != 'call_function':
            return False
        if node.target != torch.nn.functional.linear:
            if not isinstance(node.target, torch._ops.OpOverload) or node.target._schema.is_mutable:
                return False
        node = node.next
    return True


def route_fusions(gm):
    from . import _training_available
    if not _training_available:
        return gm
    changed = False
    for output in list(gm.graph.nodes):
        operands = _operands(output)
        if operands is None:
            continue
        for activation, up in (operands, operands[::-1]):
            gate = _gate(activation)
            if (gate is None or activation is up or tuple(activation.users) != (output,)
                    or not _compatible(gate, activation, up, output)
                    or not _can_delay(activation, output)):
                continue
            with gm.graph.inserting_before(output):
                fused = gm.graph.call_function(_silu_mul, (gate, up))
                fused.meta = dict(output.meta)
            output.replace_all_uses_with(fused)
            gm.graph.erase_node(output)
            gm.graph.erase_node(activation)
            changed = True
            break
    if changed:
        gm.graph.lint()
        gm.recompile()
    return gm
