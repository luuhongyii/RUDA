import math

import torch
from torch.fx import Node
from torch._subclasses.fake_tensor import FakeTensor

from ._compile_pointwise import _bind, _value


def segment_sum(value):
    width=value.shape[-1]
    mask=torch.ones((width,width),device=value.device,dtype=torch.bool)
    expanded=value.unsqueeze(-1).expand(*value.shape,width)
    scanned=expanded.masked_fill(~mask.tril(-1),0).cumsum(-2)
    return scanned.masked_fill(~mask.tril(),-torch.inf)


def segment_sum_exp(value):
    return segment_sum(value).exp()


def fuse_segment_scans(gm):
    aten=torch.ops.aten

    def call(node,*targets):
        return isinstance(node,Node) and node.op=='call_function' and node.target in targets

    def masked(node,fill):
        if not call(node,aten.masked_fill.Scalar):return None
        args=_bind(node,('self','mask','value'),{})
        if args is None or type(args['value']) not in (int,float) or args['value']!=fill:return None
        if fill==0 and math.copysign(1.,args['value'])<0:return None
        return args

    def upper_mask(node,diagonal,width,device):
        chain=[];inverted=False
        if call(node,aten.bitwise_not.default,aten.logical_not.default):
            if len(node.args)!=1 or node.kwargs:return None
            chain.append(node);node=node.args[0];inverted=True
        if not call(node,aten.tril.default if inverted else aten.triu.default):return None
        args=_bind(node,('self','diagonal'),{'diagonal':0})
        if args is None or type(args['diagonal']) is not int or args['diagonal']!=diagonal-int(inverted):return None
        chain.append(node);node=args['self'];value=_value(node)
        if (not isinstance(value,torch.Tensor) or value.dtype!=torch.bool or value.device!=device
                or any(type(d) is not int for d in value.shape) or tuple(value.shape)!=(width,width)):return None
        if call(node,aten.ones.default):
            args=_bind(node,('size','dtype','layout','device','pin_memory'),{'dtype':None,'layout':None,'device':None,'pin_memory':None})
        elif call(node,aten.full.default):
            args=_bind(node,('size','fill_value','dtype','layout','device','pin_memory'),{'dtype':None,'layout':None,'device':None,'pin_memory':None})
            if args is None or type(args['fill_value']) not in (int,float,bool) or not bool(args['fill_value']):return None
        else:return None
        if args is None or not isinstance(args['size'],(tuple,list)) or tuple(args['size'])!=(width,width):return None
        if args['layout'] not in (None,torch.strided) or args['pin_memory'] not in (None,False):return None
        chain.append(node)
        return chain

    changed=False
    for output in tuple(gm.graph.nodes):
        final=masked(output,-math.inf)
        if final is None:continue
        scan=final['self']
        if not call(scan,aten.cumsum.default):continue
        args=_bind(scan,('self','dim','dtype'),{'dtype':None})
        if args is None or args['dtype'] not in (None,torch.float32):continue
        lower=masked(args['self'],0)
        if lower is None:continue
        expanded=lower['self'];conversion=None
        if call(expanded,aten._to_copy.default):
            conversion=expanded
            cast=_bind(conversion,('self','dtype','layout','device','pin_memory','non_blocking','memory_format'),
                {'dtype':None,'layout':None,'device':None,'pin_memory':None,'non_blocking':False,'memory_format':None})
            if (cast is None or cast['dtype']!=torch.float32 or cast['layout'] not in (None,torch.strided)
                    or cast['pin_memory'] not in (None,False) or type(cast['non_blocking']) is not bool
                    or cast['memory_format'] not in (None,torch.preserve_format,torch.contiguous_format)):continue
            expanded=cast['self']
        if not call(expanded,aten.expand.default,aten.expand_copy.default):continue
        expansion=_bind(expanded,('self','size','implicit'),{'implicit':False})
        if expansion is None or type(expansion['implicit']) is not bool:continue
        unsqueezed=expansion['self']
        if not call(unsqueezed,aten.unsqueeze.default,aten.unsqueeze_copy.default):continue
        unsqueeze=_bind(unsqueezed,('self','dim'),{})
        if unsqueeze is None:continue
        source=unsqueeze['self'];value=_value(source);result=_value(output)
        if (not isinstance(value,torch.Tensor) or not isinstance(result,torch.Tensor)
                or value.device.type!='ruda' or result.device!=value.device
                or value.dtype not in (torch.float32,torch.float16,torch.bfloat16)
                or result.dtype!=torch.float32 or not result.is_contiguous()
                or not 1<=value.ndim<=7 or any(type(d) is not int for d in (*value.shape,*result.shape))
                or value.shape[-1]==0 or value.is_conj() or value.is_neg()):continue
        if conversion is None and value.dtype!=torch.float32:continue
        if conversion is not None and (not isinstance(value,FakeTensor) or cast['device'] is not None and torch.device(cast['device'])!=value.device):continue
        width=value.shape[-1];shape=(*value.shape,width)
        if (type(args['dim']) is not int or args['dim'] not in (-2,value.ndim-1)
                or type(unsqueeze['dim']) is not int or unsqueeze['dim'] not in (-1,value.ndim)
                or tuple(result.shape)!=shape or not isinstance(expansion['size'],(tuple,list))
                or len(expansion['size'])!=len(shape)
                or any(type(d) is not int or d not in (-1,size) for d,size in zip(expansion['size'],shape))):continue
        upper=upper_mask(final['mask'],1,width,value.device)
        diagonal=upper_mask(lower['mask'],0,width,value.device)
        if upper is None or diagonal is None:continue
        for part in (expanded,lower['self'],args['self'],scan):
            meta=_value(part)
            expected_dtype=value.dtype if part is expanded else torch.float32
            if (not isinstance(meta,torch.Tensor) or meta.dtype!=expected_dtype or meta.device!=value.device
                    or any(type(d) is not int for d in meta.shape) or tuple(meta.shape)!=shape):break
        else:
            end=output;target=segment_sum
            if len(output.users)==1:
                user=next(iter(output.users))
                if call(user,aten.exp.default) and user.args==(output,) and not user.kwargs:
                    meta=_value(user)
                    if (isinstance(meta,torch.Tensor) and meta.dtype==torch.float32 and meta.device==value.device
                            and tuple(meta.shape)==shape and meta.is_contiguous()):end=user;target=segment_sum_exp
            removable={output,scan,args['self'],lower['self'],expanded,unsqueezed,*upper,*diagonal}
            order={node:i for i,node in enumerate(gm.graph.nodes)}
            first=min(removable,key=order.__getitem__)
            cursor=first
            safe=True
            while cursor is not end:
                if cursor.op!='call_function' or not isinstance(cursor.target,torch._ops.OpOverload) or cursor.target._schema.is_mutable:
                    safe=False;break
                cursor=cursor.next
            if not safe:continue
            with gm.graph.inserting_before(end):
                if conversion is not None:
                    from ._compile_normalization import _Writer
                    source=_Writer(gm.graph,value).cast(source,torch.float32)
                fused=gm.graph.call_function(target,(source,));fused.meta=dict(end.meta)
            end.replace_all_uses_with(fused)
            gm.graph.erase_node(end)
            for node in sorted(removable-{end},key=order.__getitem__,reverse=True):
                if not node.users:gm.graph.erase_node(node)
            changed=True
    if changed:gm.recompile()
    return gm
