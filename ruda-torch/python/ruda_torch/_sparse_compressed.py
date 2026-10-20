import operator

import torch

from . import _C
from ._indexing import _logical
from ._ordering import _output

_libraries = []
_layouts = (torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc)
_index_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _dense(value):
    if value.layout != torch.strided or value.device.type != torch._C._get_privateuse1_backend_name():
        raise RuntimeError('compressed index conversion requires strided RUDA tensors')


def coo_to_csr(self, size, *, out_int32=False, out=None):
    _dense(self)
    size = operator.index(size)
    if self.ndim > 1 or self.dtype not in _index_dtypes:
        raise RuntimeError('COO row indices must be an integral scalar or vector')
    if size < 0 or size >= 2**32 - 1:
        raise RuntimeError('compressed dimension exceeds native U32 indexing')
    result = torch.empty((size + 1,), device=self.device, dtype=torch.int32 if out_int32 else torch.int64)
    _C.scientific_call(516, [_logical(self), result], [], [])
    return _output(result, out, (self,))


def csr_to_coo(crow_indices, col_indices, *, out_int32=False, transpose=False, out=None):
    _dense(crow_indices)
    _dense(col_indices)
    if crow_indices.device != col_indices.device or crow_indices.dtype not in _index_dtypes:
        raise RuntimeError('compressed pointers require an integral dtype and the same device as plain indices')
    if (crow_indices.ndim == 0 or crow_indices.ndim != col_indices.ndim
            or crow_indices.shape[:-1] != col_indices.shape[:-1] or crow_indices.shape[-1] == 0):
        raise RuntimeError('compressed and plain indices require matching batch dimensions and a nonempty pointer axis')
    batch_dim = col_indices.ndim - 1
    result = torch.empty((batch_dim + 2, col_indices.numel()), device=crow_indices.device,
                         dtype=torch.int32 if out_int32 else torch.int64)
    nnz = col_indices.shape[-1]
    if crow_indices.shape[-1] == 1 or nnz == 0:
        result.zero_()
    elif col_indices.numel():
        _C.scientific_call(517, [_logical(crow_indices), result], [nnz, int(transpose)], [])
        result[batch_dim + (0 if transpose else 1)].copy_(_logical(col_indices).reshape(-1))
    return _output(result, out, (crow_indices, col_indices))


def validate_indices(is_crow, compressed_idx, plain_idx, cdim, dim, nnz):
    _dense(compressed_idx)
    _dense(plain_idx)
    cdim, dim, nnz = map(operator.index, (cdim, dim, nnz))
    if (compressed_idx.device != plain_idx.device or compressed_idx.dtype != plain_idx.dtype
            or compressed_idx.dtype not in (torch.int32, torch.int64)):
        raise RuntimeError('compressed indices must have matching int32/int64 dtypes and devices')
    limit = 2**31 - 1 if compressed_idx.dtype == torch.int32 else 2**63 - 1
    if any(n < 0 or n > limit for n in (cdim, dim, nnz)):
        raise RuntimeError('compressed sparse dimensions exceed the index dtype')
    if cdim >= 2**32 - 1 or nnz >= 2**32:
        raise RuntimeError('compressed sparse index storage exceeds native U32 indexing')
    if (compressed_idx.ndim == 0 or compressed_idx.ndim != plain_idx.ndim
            or compressed_idx.shape[:-1] != plain_idx.shape[:-1]
            or compressed_idx.shape[-1] != cdim + 1 or plain_idx.shape[-1] != nnz):
        raise RuntimeError('compressed sparse index shapes do not match the batch dimensions, cdim or nnz')
    status = torch.zeros((1,), device=compressed_idx.device, dtype=torch.int32)
    _C.scientific_call(518, [_logical(compressed_idx), _logical(plain_idx), status], [cdim, dim, nnz], [])
    if int(status.item()):
        names = ('crow', 'col') if is_crow else ('ccol', 'row')
        raise RuntimeError(f'invalid {names[0]}/{names[1]} indices: pointers must start at zero and end at nnz, '
                           'with nonnegative bounded differences; plain indices must be in range, '
                           'sorted and distinct within each compressed segment')


def to_block(self, blocksize, dense_dim=None, *, column=False):
    if self.layout in (torch.sparse_bsr, torch.sparse_bsc):
        return _C.compressed_flip_block(self, blocksize, dense_dim, column)
    if self.layout not in (torch.sparse_csr, torch.sparse_csc):
        raise RuntimeError('compressed-to-block conversion requires CSR or CSC storage')
    if dense_dim is not None:
        raise RuntimeError('dense_dim is not supported when converting a compressed sparse tensor')
    blocksize = tuple(map(operator.index, blocksize))
    if len(blocksize) != 2 or any(n <= 0 for n in blocksize):
        raise RuntimeError('blocksize must contain two positive dimensions')
    if self.ndim - self.dense_dim() != 2:
        raise RuntimeError('compressed-to-block conversion does not support batched inputs')
    if self.shape[0] % blocksize[0] or self.shape[1] % blocksize[1]:
        raise RuntimeError('sparse matrix dimensions must be divisible by blocksize')
    if column and self.layout == torch.sparse_csr:
        self = self.to_sparse_csc()
    elif not column and self.layout == torch.sparse_csc:
        self = self.to_sparse_csr()
    pointers = self.ccol_indices() if column else self.crow_indices()
    plain = self.row_indices() if column else self.col_indices()
    values = _logical(self.values())
    from ._sparse_coo import _input
    _input(values, sparse=False)
    if pointers.dtype not in (torch.int32, torch.int64) or pointers.dtype != plain.dtype:
        raise RuntimeError('compressed-to-block conversion requires matching int32/int64 indices')
    compressed_block, plain_block = blocksize[::-1] if column else blocksize
    nnz = plain.numel()
    if nnz > (2**32 - 1) // 3 or compressed_block * plain_block > 2**63 - 1:
        raise RuntimeError('sparse block conversion exceeds native key storage limits')
    keys = torch.empty((nnz, 3), dtype=torch.int64, device=self.device)
    order = torch.empty((nnz,), dtype=torch.int32, device=self.device)
    groups = torch.empty_like(order)
    starts = torch.empty_like(order)
    count = torch.zeros((1,), dtype=torch.int64, device=self.device)
    blocks = 0
    if nnz:
        _C.scientific_call(519, [_logical(pointers), _logical(plain), keys, order, groups, starts, count],
                           [compressed_block, plain_block, int(column)], [])
        blocks = int(count.item())
    del groups, count
    indices = torch.empty((2, blocks), dtype=torch.int64, device=self.device)
    output_values = torch.empty((blocks, *blocksize, *values.shape[1:]), dtype=values.dtype, device=self.device)
    if blocks:
        _C.scientific_call(520, [keys, order, starts, values, indices, output_values], [], [])
    del keys, order, starts
    compressed_size = self.shape[1 if column else 0] // compressed_block
    output_pointers = coo_to_csr(indices[0], compressed_size, out_int32=pointers.dtype == torch.int32)
    output_plain = indices[1].to(plain.dtype).contiguous()
    return torch.ops.aten._sparse_compressed_tensor_unsafe.default(
        output_pointers, output_plain, output_values, list(self.shape), dtype=self.dtype,
        layout=torch.sparse_bsc if column else torch.sparse_bsr, device=self.device)


def _compressed_output(result, out):
    if result.layout not in _layouts or out.layout != result.layout or out.device != result.device:
        raise RuntimeError('compressed sparse output requires matching layout and device')
    if not torch.can_cast(result.dtype, out.dtype):
        raise RuntimeError('compressed sparse output has an incompatible dtype')
    out.resize_as_sparse_(result)
    out.copy_(result)
    return out


def _conversion_out(name):
    def implementation(*args, out, **kwargs):
        result = getattr(torch.ops.aten, name).default(*args, **kwargs)
        if result.layout == torch.strided:
            return _output(result, out, (), cast=True)
        if result.layout in _layouts:
            return _compressed_output(result, out)
        from ._sparse_coo import _sparse_output
        return _sparse_output(result, out)
    return implementation


def register(dense_library, coo_library):
    for name, function in (('_convert_indices_from_coo_to_csr', coo_to_csr),
                           ('_convert_indices_from_csr_to_coo', csr_to_coo)):
        dense_library.impl(name, function)
        dense_library.impl(name + '.out', function)
    dense_library.impl('_validate_compressed_sparse_indices', validate_indices)
    library = torch.library.Library('aten', 'IMPL', 'SparseCsrPrivateUse1')
    library.impl('_to_sparse_bsr', to_block)
    library.impl('_to_sparse_bsc', lambda self, blocksize, dense_dim=None:
                 to_block(self, blocksize, dense_dim, column=True))
    for target in (dense_library, coo_library, library):
        for name in ('_to_sparse_csr', '_to_sparse_csc', '_to_sparse_bsr', '_to_sparse_bsc'):
            target.impl(name + '.out', _conversion_out(name))
    for target in (coo_library, library):
        target.impl('_to_sparse.out', _conversion_out('_to_sparse'))
        from ._sparse_coo import _sparse_output
        target.impl('_to_sparse.sparse_dim_out', lambda self, sparse_dim, *, out:
                    _sparse_output(torch.ops.aten._to_sparse.sparse_dim(self, sparse_dim), out))
    library.impl('_to_dense.out', lambda self, dtype=None, masked_grad=None, *, out:
                 _output(torch.ops.aten._to_dense.default(self, dtype, masked_grad), out, (), cast=True))
    from ._compressed_matmul import register as register_matmul
    register_matmul(library)
    from ._sampled_matmul import register as register_sampled
    register_sampled(library)
    from ._compressed_add import register as register_add
    register_add(library)
    from ._compressed_reduce import register as register_reduce
    register_reduce(library)
    from ._spmm_reduce import register as register_mm_reduce
    register_mm_reduce(library)
    from ._sparse_triangular import register as register_triangular
    register_triangular(library)
    from ._sparse_solve import register as register_solve
    register_solve(library)
    library.impl('sparse_mask.out', _conversion_out('sparse_mask'))
    _libraries.append(library)
