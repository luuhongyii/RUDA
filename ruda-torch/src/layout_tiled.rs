use super::*;
use ruda_core::tensor::DType;

#[ruda(launch)]
fn transpose<I: Numeric, O: Numeric, V: Size>(input: &Array<Vector<I,V>>, output: &mut Array<Vector<O,V>>, rows: u32, columns: u32,
    #[define(I)] _input: StorageType, #[define(O)] _output: StorageType,
    #[define(V)] _vector: usize,
) {
    let x = UNIT_POS_X as usize;
    let y = UNIT_POS_Y as usize;
    let row_base = RUDA_POS_Y as usize * 32usize;
    let col_base = RUDA_POS_X as usize * 32usize;
    let batch = RUDA_POS_Z as usize * rows as usize * columns as usize;
    let mut tile = SharedMemory::<Vector<I,V>>::new(32usize * 33usize);
    #[unroll] for j in 0usize..4 {
        let row = row_base + y + j * 8usize;
        let column = col_base + x;
        if row < rows as usize && column < columns as usize {
            tile[(y+j*8)*33+x] = input[batch+row*columns as usize+column];
        }
    }
    sync_ruda();
    #[unroll] for j in 0usize..4 {
        let row = col_base + y + j * 8usize;
        let column = row_base + x;
        if row < columns as usize && column < rows as usize {
            output[batch+row*rows as usize+column] = Vector::cast_from(tile[x*33+y+j*8]);
        }
    }
}

fn prepare(c: &ComputeClient<CudaRuntime>, a: &View, out: &View, rows: usize, columns: usize, vector: usize)
    -> Option<PreparedKernel<CudaRuntime>>
{
    if (a.dtype > 8 && !matches!(a.dtype,12|13|15)) || (out.dtype > 8 && !matches!(out.dtype,12|13|15))
        || a.dtype != out.dtype && (a.dtype > 2 || out.dtype > 2)
        || a.len < 1024 || a.len > u32::MAX as usize || out.len != a.len || !out.contiguous()
        || rows < 2 || columns < 2
        || !matches!(vector,1|2|4|8|16) || vector*element_bytes(a.dtype).max(element_bytes(out.dtype))>16
        || a.handle.offset_start.unwrap_or(0)%(vector*element_bytes(a.dtype)) as u64!=0
        || out.handle.offset_start.unwrap_or(0)%(vector*element_bytes(out.dtype)) as u64!=0
        || a.handle.memory.clone().binding().id() == out.handle.memory.clone().binding().id()
    { return None; }
    let batches = a.len / rows.checked_mul(columns)?.checked_mul(vector)?;
    let h = &c.properties().hardware;
    if h.max_units_per_ruda < 256 || h.max_ruda_dim.0 < 32 || h.max_ruda_dim.1 < 8
        || (h.max_shared_memory_size as usize) < 32 * 33 * element_bytes(a.dtype) * vector
        || columns.div_ceil(32) > h.max_ruda_count.0 as usize || rows.div_ceil(32) > h.max_ruda_count.1 as usize
        || batches > h.max_ruda_count.2 as usize { return None; }
    let storage = |code| -> StorageType { match code {
        0=>DType::F32,1=>DType::F16,2=>DType::BF16,3|8|12|13|15=>DType::U8,
        4=>DType::I64,5=>DType::I32,6=>DType::I16,7=>DType::I8,_=>unreachable!(),
    }.into() };
    Some(transpose::prepare::<CudaRuntime>(c,
        RudaCount::Static(columns.div_ceil(32) as u32,rows.div_ceil(32) as u32,batches as u32),RudaDim::new_2d(32,8),
        a.array_arg(),out.array_arg(),rows as u32,columns as u32,storage(a.dtype),storage(out.dtype),vector))
}

pub(super) fn prepare_permute(c: &ComputeClient<CudaRuntime>, a: &View, out: &View, permutation: u32)
    -> Option<PreparedKernel<CudaRuntime>>
{
    let rank = a.shape.len();
    if rank < 2 || !a.contiguous() || a.dtype != out.dtype { return None; }
    let axis=(0..rank).find(|&d|((permutation>>(3*d))&7) as usize!=d)?;
    if axis+1>=rank {return None;}
    for d in 0..rank {
        let expected = if d == axis {axis+1} else if d == axis+1 {axis} else {d};
        if ((permutation >> (3*d)) & 7) as usize != expected { return None; }
    }
    prepare(c,a,out,a.shape[axis],a.shape[axis+1],a.shape[axis+2..].iter().product())
}

pub(super) fn prepare_copy(c: &ComputeClient<CudaRuntime>, a: &View, out: &View)
    -> Option<PreparedKernel<CudaRuntime>>
{
    let rank = a.shape.len();
    if rank < 2 {return None;}
    for axis in (0..rank-1).rev() {
        let inner=a.shape[axis+2..].iter().product::<usize>();
        if a.strides[axis]!=inner || a.strides[axis+1]!=a.shape[axis].checked_mul(inner)? {continue;}
        let mut stride=1usize;let mut valid=true;
        for d in (axis+2..rank).rev() {
            if a.shape[d]>1 && a.strides[d]!=stride {valid=false;break;}
            stride=stride.checked_mul(a.shape[d])?;
        }
        if !valid {continue;}
        stride=stride.checked_mul(a.shape[axis])?.checked_mul(a.shape[axis+1])?;
        for d in (0..axis).rev() {
            if a.shape[d]>1 && a.strides[d]!=stride {valid=false;break;}
            stride=stride.checked_mul(a.shape[d])?;
        }
        if valid {
            if let Some(kernel)=prepare(c,a,out,a.shape[axis+1],a.shape[axis],inner) {return Some(kernel);}
        }
    }
    None
}
