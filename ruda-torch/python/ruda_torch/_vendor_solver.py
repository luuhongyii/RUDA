"""cuSOLVER factorizations with GPU-only PyTorch layout/composite helpers."""
import math
import torch
from ._vendor_runtime import invoke,empty_column,integer
from . import _vendor_blas


def _copy_matrix(a):
    return empty_column(a.shape,a).copy_(a.resolve_conj().resolve_neg())


def _prefix(a):
    if a.dtype not in (torch.float32,torch.float64):
        raise ValueError('direct cuSOLVER integration requires real F32/F64')
    return 'cusolverDn'+('D' if a.dtype==torch.float64 else 'S')


def _metadata(a):
    m,n = map(integer,a.shape[-2:])
    return m,n,math.prod(a.shape[:-2]),a.element_size()


def _info(a):
    return torch.zeros(a.shape[:-2],device=a.device,dtype=torch.int32)


def lu_factor(a,pivot=True):
    factor = _copy_matrix(a)
    m,n,batches,width = _metadata(a)
    k = min(m,n)
    pivots = torch.empty((*a.shape[:-2],k),device=a.device,dtype=torch.int32)
    info = _info(a)
    if not pivot or not m or not n:
        pivots.copy_(torch.arange(1,k+1,device=a.device,dtype=torch.int32).expand(pivots.shape))
    if m and n and batches:
        invoke(2,[factor,pivots,info],[m,n,integer(batches),width,int(pivot)],prefix=_prefix(a))
    return factor,pivots,info


def lu_solve(a,pivots,b,transpose=False):
    factor,result,pivots = _copy_matrix(a),_copy_matrix(b),pivots.contiguous()
    m,n,batches,width = _metadata(a)
    nrhs = integer(b.shape[-1])
    info = _info(a)
    if m and nrhs and batches:
        invoke(3,[factor,pivots,result,info],[m,nrhs,integer(batches),width,int(transpose)],prefix=_prefix(a))
    return result,info


def cholesky(a,upper=False):
    factor = _copy_matrix(a)
    m,n,batches,width = _metadata(a)
    info = _info(a)
    if n and batches:
        invoke(4,[factor,info],[n,integer(batches),width,int(upper)],prefix=_prefix(a))
    return (torch.triu(factor) if upper else torch.tril(factor)),info


def cholesky_solve(a,b,upper=False):
    factor,result = _copy_matrix(a),_copy_matrix(b)
    m,n,batches,width = _metadata(a)
    nrhs,info = integer(b.shape[-1]),_info(a)
    if n and nrhs and batches:
        invoke(5,[factor,result,info],[n,nrhs,integer(batches),width,int(upper)],prefix=_prefix(a))
    return result,info


def geqrf(a):
    packed = _copy_matrix(a)
    m,n,batches,width = _metadata(a)
    k = min(m,n)
    tau = torch.zeros((*a.shape[:-2],k),device=a.device,dtype=a.dtype)
    info = _info(a)
    if k and batches:
        invoke(6,[packed,tau,info],[m,n,integer(batches),width],prefix=_prefix(a))
    return packed,tau,info


def _form_q(packed,tau,columns):
    m,n,batches,width = _metadata(packed)
    k,columns = integer(tau.shape[-1]),integer(columns)
    q = empty_column((*packed.shape[:-2],m,columns),packed)
    q.zero_()
    if k:
        q[...,:k].copy_(packed[...,:k])
    info = _info(packed)
    if not k:
        q.copy_(torch.eye(m,columns,device=packed.device,dtype=packed.dtype).expand(q.shape))
    elif batches and columns:
        tau = tau.contiguous()
        invoke(7,[q,tau,info],[m,columns,k,integer(batches),width],prefix=_prefix(packed))
    return q,info


def qr(a,mode=0):
    packed,tau,info = geqrf(a)
    m,n = a.shape[-2:]
    k = min(m,n)
    rows = m if mode==1 else k
    r = torch.zeros((*a.shape[:-2],rows,n),device=a.device,dtype=a.dtype)
    r[...,:min(rows,m),:].copy_(torch.triu(packed)[...,:rows,:])
    q = torch.empty((0,),device=a.device,dtype=a.dtype)
    if mode != 2:
        q,qinfo = _form_q(packed,tau,m if mode==1 else k)
        info = torch.where(info!=0,info,qinfo)
    return q,r,packed,tau,info


def ormqr(a,tau,b,transpose=False):
    packed,result,tau = _copy_matrix(a),_copy_matrix(b),tau.contiguous()
    order,columns,batches,width = _metadata(a)
    m,n = map(integer,b.shape[-2:])
    k = integer(tau.shape[-1])
    info = _info(a)
    if k and m and n and batches:
        invoke(8,[packed,tau,result,info],[order,columns,m,n,k,integer(batches),width,int(transpose)],prefix=_prefix(a))
    return result,info


def eigh(a,upper=False,compute_v=True):
    vectors = _copy_matrix(a)
    m,n,batches,width = _metadata(a)
    values = torch.empty(a.shape[:-1],device=a.device,dtype=a.dtype)
    info = _info(a)
    if n and batches:
        invoke(9,[vectors,values,info],[n,integer(batches),width,int(upper),int(compute_v)],prefix=_prefix(a))
    return values,vectors,info


def svd(a,full=False,uv=True,driver=None):
    if driver not in (None,'gesvdj','gesvd'):
        raise ValueError('NVIDIA SVD supports gesvdj and gesvd; gesvda is not substituted')
    wide = driver=='gesvd' and a.shape[-2]<a.shape[-1]
    source = a.transpose(-2,-1) if wide else a
    factor = _copy_matrix(source)
    m,n,batches,width = _metadata(source)
    k = min(m,n)
    uc,vc = (m,n) if full else (k,k)
    u = empty_column((*source.shape[:-2],m,uc),source)
    # gesvd returns VT, whereas gesvdj returns V.
    right_shape = (*source.shape[:-2],vc,n) if driver=='gesvd' else (*source.shape[:-2],n,vc)
    right = empty_column(right_shape,source)
    values = torch.empty((*source.shape[:-2],k),device=a.device,dtype=a.dtype)
    info = _info(source)
    if not k:
        if uv:
            u.copy_(torch.eye(m,uc,device=a.device,dtype=a.dtype).expand(u.shape))
            right.copy_(torch.eye(*right_shape[-2:],device=a.device,dtype=a.dtype).expand(right.shape))
    elif batches:
        invoke(11 if driver=='gesvd' else 10,[factor,values,u,right,info],
               [m,n,uc,vc,integer(batches),width,int(full),int(uv)],prefix=_prefix(source))
    vh = right if driver=='gesvd' else right.transpose(-2,-1)
    if wide:
        u,vh = vh.transpose(-2,-1),u.transpose(-2,-1)
    return u,values,vh,info


def _identity(a):
    n = a.shape[-1]
    return torch.eye(n,device=a.device,dtype=a.dtype).expand(*a.shape[:-2],n,n)


def _zero_diagonal_info(a):
    diagonal = a.diagonal(dim1=-2,dim2=-1)
    info = _info(a)
    # No host inspection of matrix data or synchronizing .item() calls.
    for i in range(diagonal.shape[-1]-1,-1,-1):
        info = torch.where(diagonal[...,i]==0,torch.full_like(info,i+1),info)
    return info


def least_squares(a,b):
    m,n,nrhs = a.shape[-2],a.shape[-1],b.shape[-1]
    if m>=n:
        packed,tau,info = geqrf(a)
        transformed,qinfo = ormqr(packed,tau,b,True)
        r = packed[...,:n,:n]
        solution = _vendor_blas.triangular(r,transformed[...,:n,:],upper=True)
        residuals = (transformed[...,n:,:]**2).sum(dim=-2) if m>n else torch.empty((0,),device=a.device,dtype=a.dtype)
    else:
        packed,tau,info = geqrf(a.transpose(-2,-1))
        r = packed[...,:m,:m]
        y = _vendor_blas.triangular(r,b,upper=True,transpose=True)
        padded = torch.zeros((*a.shape[:-2],n,nrhs),device=a.device,dtype=a.dtype)
        padded[...,:m,:].copy_(y)
        solution,qinfo = ormqr(packed,tau,padded,False)
        residuals = torch.empty((0,),device=a.device,dtype=a.dtype)
    info = torch.where(info!=0,info,qinfo)
    singular = _zero_diagonal_info(r)
    info = torch.where(info!=0,info,singular)
    return solution,residuals,info


def command(op,tensors,ints=()):
    a = tensors[0]
    flag = lambda i,default=False: bool(ints[i]) if i<len(ints) else default
    if op==0:
        values = lu_factor(a,flag(0,True))
    elif op==1:
        values = lu_solve(a,tensors[1],tensors[2],flag(0))
    elif op==2:
        values = cholesky(a,flag(0))
    elif op==3:
        values = (_vendor_blas.triangular(a,tensors[1],upper=flag(0,True),transpose=flag(1),unit=flag(2)),_info(a))
    elif op==4:
        values = cholesky_solve(a,tensors[1],flag(0))
    elif op==5:
        values = qr(a,ints[0] if ints else 0)
    elif op==6:
        values = least_squares(a,tensors[1])
    elif op==7:
        eigenvalues,vectors,info = eigh(a,flag(0),flag(1,True))
        values = eigenvalues,vectors if flag(1,True) else tensors[2],info
    elif op==8:
        u,s,vh,info = svd(a,flag(0),flag(1,True),'gesvd' if len(ints)>2 and ints[2]==1 else None)
        values = (u if flag(1,True) else tensors[1],s,vh if flag(1,True) else tensors[3],info)
    elif op==9:
        lu,piv,info = lu_factor(a)
        inverse,_ = lu_solve(lu,piv,_identity(a))
        values = inverse,lu,piv,info
    elif op==10:
        lu,piv,info = lu_factor(a)
        diagonal = lu.diagonal(dim1=-2,dim2=-1)
        indices = torch.arange(1,piv.shape[-1]+1,device=a.device,dtype=torch.int32)
        parity = ((piv!=indices).to(torch.int32).sum(dim=-1)%2).to(a.dtype)
        permutation_sign = 1-2*parity
        det = diagonal.prod(dim=-1)*permutation_sign
        sign = diagonal.sign().prod(dim=-1)*permutation_sign
        logabs = diagonal.abs().log().sum(dim=-1)
        values = lu,piv,det,sign,logabs
    elif op==11:
        result,_ = ormqr(a,tensors[1],tensors[2],flag(0))
        values = (result,)
    elif op==12:
        values = geqrf(a)
    elif op==13:
        inverse,info = cholesky_solve(a,_identity(a),flag(0))
        singular = _zero_diagonal_info(a)
        values = inverse,torch.where(info!=0,info,singular)
    elif op==14:
        result,_ = _form_q(a,tensors[1],a.shape[-1])
        values = (result,)
    else:
        raise ValueError(f'unsupported NVIDIA linalg command {op}')
    begin = {0:1,1:3,2:1,3:2,4:2,5:1,6:2,7:1,8:1,9:1,10:1,11:3,12:1,13:1,14:2}[op]
    targets = tensors[begin:]
    if len(values)!=len(targets):
        raise RuntimeError('NVIDIA linalg output contract mismatch')
    for target,value in zip(targets,values):
        if target is not value:
            target.copy_(value)
