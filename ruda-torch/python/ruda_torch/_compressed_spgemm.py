import torch

from . import _C
from ._indexing import _logical
from .math_backend import _nvidia

_layouts = (torch.sparse_csr, torch.sparse_csc)
_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128)


def _matrix(value):
    if (value.layout not in _layouts or value.ndim != 2 or value.dense_dim() != 0
            or value.device.type != torch._C._get_privateuse1_backend_name() or value.dtype not in _dtypes):
        raise RuntimeError('sparse-sparse matrix multiplication requires non-batched CSR/CSC floating or complex RUDA matrices')
    if max(*value.shape, value._nnz()) > 2**31 - 1:
        raise RuntimeError('compressed sparse matrix multiplication exceeds int32 index limits')


def _indices(value):
    if value.layout == torch.sparse_csr:
        return value.crow_indices(), value.col_indices()
    return value.ccol_indices(), value.row_indices()


def _make(pointers, indices, values, shape, layout):
    return torch.ops.aten._sparse_compressed_tensor_unsafe.default(
        pointers, indices, values, list(shape), dtype=values.dtype, layout=layout, device=values.device)


def _empty(reference, shape, layout, index_dtype):
    axis = 0 if layout == torch.sparse_csr else 1
    pointers = torch.zeros((shape[axis] + 1,), dtype=index_dtype, device=reference.device)
    indices = torch.empty((0,), dtype=index_dtype, device=reference.device)
    values = torch.empty((0,), dtype=reference.dtype, device=reference.device)
    return _make(pointers, indices, values, shape, layout)


def _finish(result, out):
    if out is None:
        return result
    _C.compressed_mm_output(out, result)
    return out


def _native(left, right, bias, coefficients):
    left = left.to_sparse_csr() if left.layout == torch.sparse_csc else left
    right = right.to_sparse_csr() if right.layout == torch.sparse_csc else right
    rows = left.shape[0]
    ap, aj = (_logical(t).to(torch.int64) for t in _indices(left))
    bp, bj = (_logical(t).to(torch.int64) for t in _indices(right))
    if coefficients[2] == 0 and coefficients[3] == 0:
        cp = torch.zeros((rows + 1,), dtype=torch.int64, device=left.device)
        cj = torch.empty((0,), dtype=torch.int64, device=left.device)
        cv = torch.empty((0,), dtype=left.dtype, device=left.device)
    else:
        bias = bias.to_sparse_csr() if bias.layout == torch.sparse_csc else bias
        cp, cj = (_logical(t).to(torch.int64) for t in _indices(bias))
        cv = _logical(bias.values())
    heap = torch.empty((left._nnz(),), dtype=torch.int32, device=left.device)
    cursors, ends = torch.empty_like(heap), torch.empty_like(heap)
    counts = torch.zeros((rows + 1,), dtype=torch.int64, device=left.device)
    _C.scientific_call(523, [ap, aj, bp, bj, cp, cj, heap, cursors, ends, counts], [], [])
    pointers = counts.cumsum(0, dtype=torch.int64)
    del counts
    nnz = int(pointers[-1].item())
    if not 0 <= nnz <= 2**31 - 1:
        raise RuntimeError('compressed sparse matrix product exceeds int32 index limits')
    indices = torch.empty((nnz,), dtype=torch.int32, device=left.device)
    values = torch.empty((nnz,), dtype=left.dtype, device=left.device)
    if nnz:
        _C.scientific_call(524, [ap, aj, bp, bj, cp, cj, heap, cursors, ends, pointers,
                               _logical(left.values()), _logical(right.values()), cv, indices, values], [], coefficients)
    return _make(pointers.to(torch.int32), indices, values, (rows, right.shape[1]), torch.sparse_csr)


def multiply(mat1, mat2, *, bias=None, beta=0, alpha=1, out=None, inplace=False):
    _matrix(mat1)
    _matrix(mat2)
    if mat1.dtype != mat2.dtype or mat1.device != mat2.device or mat1.shape[1] != mat2.shape[0]:
        raise RuntimeError('sparse-sparse matrix multiplication requires matching dtypes, devices and inner dimensions')
    shape = (mat1.shape[0], mat2.shape[1])
    if bias is not None:
        _matrix(bias)
        if bias.dtype != mat1.dtype or bias.device != mat1.device or bias.shape != shape:
            raise RuntimeError('sparse addmm bias must have matching dtype, device and exact result shape')
    if inplace:
        out = bias
    layout = bias.layout if bias is not None else torch.sparse_csr
    if out is not None:
        if (out.layout not in _layouts or out.dtype != mat1.dtype or out.device != mat1.device
                or out.ndim != 2 or out.dense_dim() != 0):
            raise RuntimeError('sparse-sparse matrix output must be a non-batched CSR/CSC tensor with matching dtype and device')
        if bias is not None and out.layout != bias.layout:
            raise RuntimeError('sparse addmm bias and output must have matching layouts')
        layout = out.layout
    if layout == torch.sparse_csc and (mat1.layout != layout or mat2.layout != layout):
        raise RuntimeError('CSC matrix output requires two CSC multiplicands')
    if _nvidia():
        raise NotImplementedError('explicit NVIDIA sparse matmul requires a cuSPARSE route; native RUDA is not an implicit fallback')
    if not mat1._nnz() or not mat2._nnz() or 0 in shape:
        index_dtype = _indices(out)[0].dtype if out is not None else torch.int64
        if bias is None or beta == 0:
            result = _empty(mat1, shape, layout, index_dtype)
        else:
            pointers, indices = (_logical(t).to(index_dtype).clone() for t in _indices(bias))
            values = _logical(bias.values()).clone()
            if beta != 1:
                values.mul_(beta)
            result = _make(pointers, indices, values, shape, layout)
        return _finish(result, out)
    coefficients, _ = _C.blas_accumulate_scalars(alpha, beta, mat1.dtype, True)
    if layout == torch.sparse_csc:
        result = _native(mat2.transpose(0, 1), mat1.transpose(0, 1),
                         None if bias is None else bias.transpose(0, 1), coefficients).transpose(0, 1)
    else:
        result = _native(mat1, mat2, bias, coefficients)
    return _finish(result, out)
