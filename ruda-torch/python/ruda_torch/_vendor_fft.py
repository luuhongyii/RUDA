"""Direct cuFFT plans on the RUDA queue, including n-D separable transforms."""
import math
from threading import RLock
import torch
from ._vendor_runtime import invoke


def _resize(input, shape):
    if tuple(input.shape) == tuple(shape):
        return input
    output = torch.zeros(tuple(shape), device=input.device, dtype=input.dtype)
    slices = tuple(slice(0,min(a,b)) for a,b in zip(input.shape,shape))
    output[slices].copy_(input[slices])
    return output


def _scale(kind, norm, lengths):
    if kind in (4,5) and norm in (0,1):
        norm = 1-norm
    total = math.prod(lengths)
    inverse = kind in (1,3,4)
    if norm == 2:
        return 1/math.sqrt(total)
    if norm == 0 and inverse or norm == 1 and not inverse:
        return 1/total
    return 1.


class _AxisPlan:
    def __init__(self, length, batches, dtype, kind, stream):
        self.length, self.batches, self.kind, self.stream = length,batches,kind,stream
        self.double = dtype in (torch.float64,torch.complex128)
        self.handle, self.workspace, self.bytes = 0,0,0
        self._create()

    def _create(self):
        if self.handle or not self.batches:
            return
        self.handle,self.workspace,self.bytes = invoke(12,[],
            [self.length,self.batches,int(self.double),self.kind],stream=self.stream)

    def execute(self, input, axis):
        self._create()
        order = tuple(i for i in range(input.ndim) if i != axis)+(axis,)
        inverse = tuple(order.index(i) for i in range(input.ndim))
        target = list(input.shape)
        target[axis] = self.length//2+1 if self.kind == 3 else self.length
        packed = _resize(input,target).permute(order).contiguous()
        out_shape = (*packed.shape[:-1], self.length//2+1 if self.kind == 2 else self.length)
        dtype = (torch.float64 if self.double else torch.float32) if self.kind == 3 else (
            torch.complex128 if self.double else torch.complex64)
        output = torch.empty(out_shape,device=input.device,dtype=dtype)
        if not self.batches:
            return output.permute(inverse)
        invoke(13,[packed,output],[self.handle,self.kind,int(self.double)],stream=self.stream)
        return output.permute(inverse)

    def clear_workspace(self):
        if not self.handle:
            return
        from ._streams import Stream, stream
        with stream(Stream(_id=self.stream)):
            invoke(14,[],[self.handle,self.workspace],stream=self.stream)
        self.handle,self.workspace,self.bytes = 0,0,0


class VendorFFTPlan:
    vendor = True
    def __init__(self, input, kind, axes, lengths, norm):
        from . import _C
        self.shape,self.dtype,self.device = tuple(input.shape),input.dtype,input.device
        self.kind,self.axes,self.lengths,self.norm = kind,tuple(axes),tuple(lengths),norm
        self.stream = _C.stream_command(0)
        self._lock,self.closed = RLock(),False
        self.steps = []
        # cuFFT rank-1 batching also covers >3 transformed dimensions and any
        # axis order, without restricting the PyTorch n-D contract.
        if kind in (2,5):
            steps = [(axes[-1],lengths[-1],2)]+[(a,n,0) for a,n in zip(axes[:-1],lengths[:-1])]
        elif kind in (3,4):
            steps = [(a,n,1) for a,n in zip(axes[:-1],lengths[:-1])]+[(axes[-1],lengths[-1],3)]
        else:
            steps = [(a,n,kind) for a,n in zip(axes,lengths)]
        shape,dtype = list(input.shape),input.dtype
        try:
            for axis,length,step in steps:
                batches = math.prod(shape[:axis]+shape[axis+1:])
                self.steps.append((axis,_AxisPlan(length,batches,dtype,step,self.stream)))
                shape[axis] = length//2+1 if step == 2 else length
                dtype = (torch.float64 if dtype in (torch.float64,torch.complex128) else torch.float32) if step == 3 else (
                    torch.complex128 if dtype in (torch.float64,torch.complex128) else torch.complex64)
        except BaseException:
            self.close()
            raise

    def execute(self, input, output):
        with self._lock:
            if self.closed:
                raise RuntimeError('cuFFT plan is closed')
            if (tuple(input.shape),input.dtype,input.device) != (self.shape,self.dtype,self.device):
                raise ValueError('cuFFT plan input geometry changed')
            from . import _C
            if _C.stream_command(0) != self.stream:
                raise RuntimeError('cuFFT plan is bound to its creation RUDA stream')
            value = input.resolve_conj().resolve_neg()
            if self.kind == 4:
                value = value.conj().resolve_conj()
            if self.kind in (0,1) and not value.is_complex():
                value = value.to(torch.complex128 if value.dtype == torch.float64 else torch.complex64)
            for axis,plan in self.steps:
                value = plan.execute(value,axis)
            if self.kind == 5:
                value = value.conj().resolve_conj()
            scale = _scale(self.kind,self.norm,self.lengths)
            if scale != 1:
                value = value*scale
            output.copy_(value)

    def retained_bytes(self):
        with self._lock:
            return sum(plan.bytes for _,plan in self.steps)

    def clear_workspace(self):
        with self._lock:
            if self.closed:
                raise RuntimeError('cuFFT plan is closed')
            for _,plan in self.steps:
                plan.clear_workspace()

    def close(self):
        with self._lock:
            if not self.closed:
                for _,plan in self.steps:
                    plan.clear_workspace()
                self.closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def transform(input,kind,axes,lengths,norm,output):
    plan = VendorFFTPlan(input,kind,axes,lengths,norm)
    try:
        plan.execute(input,output)
    finally:
        plan.close()


def _complex_transform(input,axes,lengths,inverse):
    shape = list(input.shape)
    for axis,length in zip(axes,lengths):
        shape[axis] = length
    dtype = torch.complex128 if input.dtype in (torch.float64,torch.complex128) else torch.complex64
    output = torch.empty(shape,device=input.device,dtype=dtype)
    transform(input,1 if inverse else 0,axes,lengths,3,output)
    return output


def adjoint(grad,kind,axes,lengths,norm,shape,dtype,output):
    grad = grad.resolve_conj().resolve_neg()
    scale = _scale(kind,norm,lengths)
    last = axes[-1]
    if kind in (0,1):
        result = _complex_transform(grad,axes,lengths,kind==0)
    elif kind in (2,5):
        if kind == 5:
            grad = grad.conj().resolve_conj()
        full = list(grad.shape)
        for axis,length in zip(axes,lengths):
            full[axis] = length
        # R2C's adjoint is zero extension, not C2R's doubled Hermitian extension.
        result = _complex_transform(_resize(grad,full),axes,lengths,True).real
    else:
        result = _complex_transform(grad,axes,lengths,False)
        result = result.narrow(last,0,lengths[-1]//2+1).clone()
        interior = (lengths[-1]-1)//2
        if interior:
            result.narrow(last,1,interior).mul_(2)
        if kind == 4:
            result = result.conj().resolve_conj()
    if dtype in (torch.float32,torch.float64) and result.is_complex():
        result = result.real
    result = _resize(result,shape)
    if scale != 1:
        result = result*scale
    output.copy_(result.to(dtype))
