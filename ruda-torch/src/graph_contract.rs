//! Host-only validation shared by the native static graph bridge.
//! No backend calls or GPU emulation. Kernels use fixed, contiguous buffers; AOTAutograd owns training differentiation.

pub const MAX_NODES: usize = 256;
pub const MAX_TENSORS: usize = 512;
pub const NO_WEIGHT: u32 = u32::MAX;
pub const COPY: u32 = 0;
pub const ADD: u32 = 1;
pub const MUL: u32 = 2;
pub const SILU: u32 = 14;
pub const RMS_NORM: u32 = 100;
pub const SILU_MUL: u32 = 101;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NodeSpec {
    pub op: u32,
    pub a: u32,
    pub b: u32,
    pub scalar: f32,
}

#[derive(Debug)]
pub struct TensorSpec<'a> {
    pub shape: &'a [usize],
    pub strides: &'a [usize],
    pub dtype: u32,
}

fn broadcast_shape(inputs:&[&[usize]],out:&[usize])->bool {
    let rank=inputs.iter().map(|shape|shape.len()).max().unwrap_or(0);
    if rank!=out.len() {return false;}
    for d in 0..rank {
        let mut size=1;
        for shape in inputs {
            let value=if d+shape.len()<rank {1} else {shape[d+shape.len()-rank]};
            if size!=1 && value!=1 && size!=value {return false;}if size==1 {size=value;}
        }
        if out[d]!=size {return false;}
    }
    true
}

fn scalar_fits(dtype:u32,value:f32)->bool {
    match dtype {
        0|3=>true,
        1=>!value.is_finite() || value.abs()<=65504.0,
        2=>!value.is_finite() || value.abs()<=f32::from_bits(0x7f7f0000),
        4=>value.is_finite() && value.fract()==0.0 && (value as f64)>=i64::MIN as f64 && (value as f64)<9223372036854775808.0,
        5=>value.is_finite() && value.fract()==0.0 && (value as f64)>=i32::MIN as f64 && (value as f64)<=i32::MAX as f64,
        6=>value.is_finite() && value.fract()==0.0 && value>=i16::MIN as f32 && value<=i16::MAX as f32,
        7=>value.is_finite() && value.fract()==0.0 && value>=i8::MIN as f32 && value<=i8::MAX as f32,
        8=>value.is_finite() && value.fract()==0.0 && value>=0.0 && value<=255.0,
        _=>false,
    }
}

pub fn validate(tensors: &[TensorSpec<'_>], nodes: &[NodeSpec], inputs: usize) -> Result<(), String> {
    if nodes.is_empty() || nodes.len() > MAX_NODES
        || tensors.len() > MAX_TENSORS || inputs.checked_add(nodes.len()) != Some(tensors.len()) {
        return Err("static graph requires 1..256 nodes and one logical output per node (max 512 tensors)".into());
    }
    for t in tensors {
        if t.dtype > 8 || t.shape.len() > 8 || t.shape.len() != t.strides.len() {
            return Err("static graph accepts rank 0..8 floating/integer/bool tensors".into());
        }
        if t.shape.iter().any(|&dim|dim>u32::MAX as usize) {
            return Err("static graph dimension exceeds 32-bit kernel indexing".into());
        }
        if t.shape.contains(&0) {continue;}
        let mut count = 1usize;
        for (&dim, &stride) in t.shape.iter().zip(t.strides).rev() {
            if dim > 1 && stride != count {
                return Err("static graph requires contiguous storage".into());
            }
            count = count.checked_mul(dim).ok_or("static graph shape overflow")?;
        }
        if count > u32::MAX as usize { return Err("static graph exceeds 32-bit kernel indexing".into()); }
    }
    for (i, n) in nodes.iter().enumerate() {
        let out_index = inputs + i;
        let out = &tensors[out_index];
        let root=matches!(n.op,160|161|172) && n.a==NO_WEIGHT;
        if !root && n.a as usize >= out_index { return Err("static graph input must precede its output".into()); }
        let a = if root {out} else {&tensors[n.a as usize]};
        if (!matches!(n.op,114|147..=161|164..=167|172) && a.dtype != out.dtype) || (!matches!(n.op, 7 | 30 | 57 | 104 | 105 | 112..=124 | 134..=172) && a.shape != out.shape) {
            return Err("static graph output shape/dtype mismatch".into());
        }
        if (a.dtype>2 || out.dtype>2) && !matches!(n.op,0|112..=115|134..=136|140..=172) {
            return Err("nonfloating graph tensors require a storage/layout/indexing operation".into());
        }
        match n.op {
            172 => {
                if out.dtype>2 || out.shape.len()!=1 || !n.scalar.is_finite() || !f32::from_bits(n.b).is_finite() {
                    return Err("invalid floating arange graph parameters".into());
                }
            }
            162..=171 => {
                if n.b as usize>=out_index || !n.scalar.is_finite() {return Err("invalid mask graph input/scalar".into());}
                let b=&tensors[n.b as usize];
                if n.op<=163 {
                    if a.shape.len()<2 || n.b!=n.a || a.shape!=out.shape || n.scalar.fract()!=0.0
                        || (n.scalar as f64)<i32::MIN as f64 || (n.scalar as f64)>i32::MAX as f64 {return Err("invalid triangular graph mask".into());}
                } else {
                    if n.scalar!=0.0 || a.dtype!=b.dtype || !broadcast_shape(&[a.shape,b.shape],out.shape)
                        || (matches!(n.op,164|168) && n.b!=n.a) {return Err("invalid logical/bitwise graph operands".into());}
                    if n.op<=167 {if out.dtype!=3 {return Err("logical graph output must be bool".into());}}
                    else if a.dtype<3 {return Err("bitwise graph operands must be integer/bool".into());}
                }
            }
            160|161 => {
                if !scalar_fits(out.dtype,n.scalar) {return Err("factory scalar exceeds output dtype".into());}
                if n.op==160 {
                    if n.b!=n.a {return Err("full requires one device anchor".into());}
                } else {
                    if out.shape.len()!=1 || !(4..=8).contains(&out.dtype) || n.b as i32==0 {
                        return Err("arange_integer requires an integer vector and nonzero signed step".into());
                    }
                    let end=n.scalar as i64 as i128+out.shape[0].saturating_sub(1) as i128*(n.b as i32 as i128);
                    let (low,high)=match out.dtype {
                        4=>(i64::MIN as i128,i64::MAX as i128),5=>(i32::MIN as i128,i32::MAX as i128),
                        6=>(i16::MIN as i128,i16::MAX as i128),7=>(i8::MIN as i128,i8::MAX as i128),8=>(0,255),_=>unreachable!(),
                    };
                    if end<low || end>high {return Err("arange_integer endpoint exceeds output dtype".into());}
                }
            }
            146..=159 => {
                if n.b as usize>=out_index {return Err("predicate graph input must precede output".into());}
                let b=&tensors[n.b as usize];
                if n.op==146 {
                    if b.dtype!=3 || !scalar_fits(a.dtype,n.scalar) || !broadcast_shape(&[a.shape,b.shape],out.shape) {
                        return Err("invalid graph masked_fill dtype/broadcast/scalar".into());
                    }
                } else if n.op==147 {
                    if !n.scalar.is_finite() || n.scalar.fract()!=0.0 || n.scalar<0.0 || n.scalar as usize>=out_index {
                        return Err("where third input must precede output".into());
                    }
                    let c=&tensors[n.scalar as usize];
                    if a.dtype!=3 || b.dtype!=c.dtype || out.dtype!=b.dtype || !broadcast_shape(&[a.shape,b.shape,c.shape],out.shape) {
                        return Err("invalid graph where dtype/broadcast".into());
                    }
                } else {
                    if out.dtype!=3 {return Err("graph comparison output must be bool".into());}
                    if n.op<154 {
                        if n.scalar.to_bits()!=0 || a.dtype!=b.dtype || !broadcast_shape(&[a.shape,b.shape],out.shape) {return Err("invalid graph comparison operands".into());}
                    } else if n.b!=n.a || out.shape!=a.shape || !scalar_fits(a.dtype,n.scalar) || (a.dtype==3 && n.scalar!=0.0 && n.scalar!=1.0) {
                        return Err("invalid graph scalar comparison".into());
                    }
                }
            }
            140..=145 => {
                if n.b as usize>=out_index || !n.scalar.is_finite() || n.scalar<0.0 || n.scalar.fract()!=0.0 || n.scalar>=16777216.0 {
                    return Err("invalid graph slice/cat encoding".into());
                }
                let code=n.scalar as usize;let axis=if n.op==142 {code} else {code&7};let start=code>>3;
                let b=&tensors[n.b as usize];
                if b.dtype!=a.dtype {return Err("slice/cat requires equal storage dtype".into());}
                if n.op==142 {
                    if axis>=a.shape.len() || b.shape.len()!=a.shape.len() || out.shape.len()!=a.shape.len()
                        || a.shape.iter().enumerate().any(|(d,&size)|if d==axis {size.checked_add(b.shape[d])!=Some(out.shape[d])} else {size!=b.shape[d] || size!=out.shape[d]}) {
                        return Err("invalid two-input graph concatenation".into());
                    }
                } else {
                    let (full,part)=if n.op==143 || n.op==144 {(out,a)} else if n.op==145 {(a,b)} else {(a,out)};
                    let selected=n.op==141 || n.op==144;
                    if axis>=full.shape.len() || (n.op!=145 && n.a!=n.b) || (n.op==145 && a.shape!=out.shape) {
                        return Err("invalid graph slice operands/axis".into());
                    }
                    let mut expected=full.shape.to_vec();
                    if selected {if start>=expected[axis] {return Err("graph select out of bounds".into());}expected.remove(axis);}
                    else {
                        if part.shape.len()!=full.shape.len() || start.checked_add(part.shape[axis]).is_none_or(|end|end>full.shape[axis]) {return Err("graph slice out of bounds".into());}
                        expected[axis]=part.shape[axis];
                    }
                    if part.shape!=expected.as_slice() {return Err("graph slice output shape differs".into());}
                }
            }
            134..=139 => {
                if n.b as usize>=out_index || !n.scalar.is_finite() || n.scalar.fract()!=0.0 {
                    return Err("invalid indexing graph input/scalar".into());
                }
                let b=&tensors[n.b as usize];
                if !matches!(b.dtype,4|5) {return Err("graph indices require int32/int64 storage".into());}
                if matches!(n.op,136..=138) {
                    let (weight,gradient)=if n.op==136 {(a,out)} else {(out,a)};
                    let mut expected=b.shape.to_vec();
                    if weight.shape.len()!=2 {return Err("graph embedding weight must be a matrix".into());}
                    expected.push(weight.shape[1]);
                    if gradient.shape!=expected.as_slice() || (n.op==136 && n.scalar.to_bits()!=0)
                        || (n.op!=136 && (n.scalar< -1.0 || n.scalar>=weight.shape[0] as f32)) {
                        return Err("graph embedding shape or padding differs".into());
                    }
                } else {
                    let (source,indexed)=if n.op==139 {(out,a)} else {(a,out)};
                    if n.scalar<0.0 || n.scalar as usize>=source.shape.len() || indexed.shape.len()!=source.shape.len() {
                        return Err("invalid graph indexing axis/rank".into());
                    }
                    let axis=n.scalar as usize;
                    if n.op==134 {
                        if b.shape.len()>1 || indexed.shape.iter().enumerate().any(|(d,&size)|size!=if d==axis {b.shape.iter().product()} else {source.shape[d]}) {
                            return Err("invalid graph index_select shape".into());
                        }
                    } else if b.shape!=indexed.shape || b.shape.iter().enumerate().any(|(d,&size)|d!=axis && size>source.shape[d]) {
                        return Err("invalid graph gather shape".into());
                    }
                }
            }
            112 | 114 => {
                if n.b != n.a || n.scalar.to_bits()!=0 || (n.op==112 && a.shape.iter().product::<usize>()!=out.shape.iter().product::<usize>())
                    || (n.op==114 && a.shape!=out.shape) {
                    return Err("invalid graph reshape/cast".into());
                }
            }
            113 => {
                let code=n.scalar as u32;
                if !n.scalar.is_finite() || code as f32!=n.scalar || n.b!=n.a || a.shape.len()!=out.shape.len()
                    || code >= (1u32 << (3*a.shape.len())) {
                    return Err("invalid graph permutation encoding".into());
                }
                let mut seen=0u32;
                for (dim,&size) in out.shape.iter().enumerate() {
                    let axis=((code>>(3*dim))&7) as usize;
                    if axis>=a.shape.len() || seen&(1<<axis)!=0 || size!=a.shape[axis] {
                        return Err("invalid graph permutation axes/shape".into());
                    }
                    seen|=1<<axis;
                }
            }
            115 => {
                if n.b!=n.a || n.scalar.to_bits()!=0 || a.shape.len()>out.shape.len()
                    || a.shape.iter().rev().zip(out.shape.iter().rev()).any(|(&x,&y)|x!=1 && x!=y) {
                    return Err("invalid graph expansion".into());
                }
            }
            57 | 116..=118 | 121..=124 => {
                if n.b as usize>=out_index || !n.scalar.is_finite() || (n.op!=116 && n.scalar.to_bits()!=0) {
                    return Err("invalid graph broadcast inputs/scalar".into());
                }
                let b=&tensors[n.b as usize];
                let rank=a.shape.len().max(b.shape.len());
                if b.dtype!=a.dtype || out.shape.len()!=rank {return Err("invalid graph broadcast dtype/rank".into());}
                for dim in 0..rank {
                    let x=if dim+a.shape.len()<rank {1} else {a.shape[dim+a.shape.len()-rank]};
                    let y=if dim+b.shape.len()<rank {1} else {b.shape[dim+b.shape.len()-rank]};
                    if (x!=y && x!=1 && y!=1) || out.shape[dim]!=(if x==1 {y} else {x}) {
                        return Err("invalid graph broadcast output".into());
                    }
                }
            }
            119 | 120 => {
                let mask=n.scalar as u32;
                let expected:Vec<_>=a.shape.iter().enumerate().filter_map(|(d,&size)|
                    if mask&(1<<d)==0 {Some(size)} else {None}).collect();
                if !n.scalar.is_finite() || mask as f32!=n.scalar || mask==0 || mask>=(1<<a.shape.len())
                    || n.b!=n.a || out.shape!=expected.as_slice() {
                    return Err("invalid squeezed reduction output/mask".into());
                }
            }
            COPY | 3 | 9..=14 | 17 | 19..=25 | 27 | 35..=40 | 43 | 45 | 108 => {
                if n.b != n.a || n.scalar.to_bits() != 0f32.to_bits() {
                    return Err("unary graph node must use its input as dummy and canonical zero scalar".into());
                }
            }
            ADD | MUL | 4 | 8 | 15 | 16 | 18 | SILU_MUL => {
                if n.b as usize >= out_index { return Err("static graph second input must precede its output".into()); }
                let b = &tensors[n.b as usize];
                if a.shape != b.shape || a.dtype != b.dtype { return Err("static graph broadcasting/type promotion is not supported".into()); }
                if !n.scalar.is_finite() || (!matches!(n.op, ADD | 4) && n.scalar.to_bits() != 0f32.to_bits()) {
                    return Err("invalid static graph pointwise scalar".into());
                }
            }
            109..=111 | 129..=133 => {
                if n.b != n.a || !n.scalar.is_finite() {
                    return Err("invalid unary scalar operation".into());
                }
            }
            7 | 30 => {
                if n.b as usize >= out_index { return Err("future matrix input".into()); }
                let b = &tensors[n.b as usize];
                let rank = if n.op == 7 { 2 } else { 3 };
                if a.shape.len() != rank || b.shape.len() != rank || out.shape.len() != rank
                    || b.dtype != a.dtype || n.scalar.to_bits() != 0
                    || a.shape[rank-1] != b.shape[rank-2]
                    || out.shape[rank-2] != a.shape[rank-2] || out.shape[rank-1] != b.shape[rank-1]
                    || a.shape[..rank-2] != b.shape[..rank-2] || a.shape[..rank-2] != out.shape[..rank-2] {
                    return Err("invalid static matrix multiply shapes/dtypes".into());
                }
            }
            102 | 103 | 106 | 107 => {
                if !n.scalar.is_finite() || n.scalar.fract() != 0.0 || n.scalar < 0.0
                    || n.scalar as usize >= a.shape.len() || n.b as usize >= out_index {
                    return Err("invalid softmax axis or input".into());
                }
                let b = &tensors[n.b as usize];
                if b.shape != a.shape || b.dtype != a.dtype || (n.op < 106 && n.a != n.b) {
                    return Err("softmax gradient/output mismatch".into());
                }
            }
            104 | 105 => {
                let mask = n.scalar as u32;
                if !n.scalar.is_finite() || mask as f32 != n.scalar || mask == 0
                    || mask >= (1 << a.shape.len()) || n.b != n.a || out.shape.len() != a.shape.len()
                    || a.shape.iter().enumerate().any(|(d, &v)| out.shape[d] != if mask & (1 << d) != 0 {1} else {v}) {
                    return Err("invalid keepdim reduction mask/output".into());
                }
            }
            RMS_NORM => {
                if !n.scalar.is_finite() || n.scalar <= 0.0 { return Err("RMSNorm epsilon must be finite and positive".into()); }
                if a.shape.is_empty() {return Err("RMSNorm requires a feature axis".into());}
                let width = *a.shape.last().unwrap();
                let rows = if width==0 {0} else {a.shape.iter().product::<usize>() / width};
                if rows > u32::MAX as usize / 32 { return Err("RMSNorm row grid exceeds 32-bit indexing".into()); }
                if n.b != NO_WEIGHT {
                    if n.b as usize >= out_index { return Err("RMSNorm weight must precede its output".into()); }
                    let w = &tensors[n.b as usize];
                    if w.shape != [width] || w.dtype != a.dtype { return Err("RMSNorm requires a same-dtype last-axis weight".into()); }
                }
            }
            _ => return Err("unsupported static graph operator (no CPU fallback)".into()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(shape: &'static [usize], strides: &'static [usize], dtype: u32) -> TensorSpec<'static> {
        TensorSpec {shape, strides, dtype}
    }
    fn add() -> NodeSpec { NodeSpec {op: ADD, a: 0, b: 1, scalar: 1.0} }
    #[test] fn valid_add() { assert!(validate(&[t(&[2,3], &[3,1],0),t(&[2,3], &[3,1],0),t(&[2,3], &[3,1],0)], &[add()],2).is_ok()); }
    #[test] fn empty_graph() { assert!(validate(&[], &[],0).is_err()); }
    #[test] fn wrong_slot_count() { assert!(validate(&[t(&[1],&[1],0)], &[add()],1).is_err()); }
    #[test] fn forward_reference() { let mut n=add(); n.b=2; assert!(validate(&[t(&[1],&[1],0),t(&[1],&[1],0),t(&[1],&[1],0)], &[n],2).is_err()); }
    #[test] fn zero_dimension() { assert!(validate(&[t(&[0],&[1],0),t(&[0],&[1],0)], &[NodeSpec{op:COPY,a:0,b:0,scalar:0.}],1).is_err()); }
    #[test] fn noncontiguous() { assert!(validate(&[t(&[2],&[2],0),t(&[2],&[1],0)], &[NodeSpec{op:COPY,a:0,b:0,scalar:0.}],1).is_err()); }
    #[test] fn no_promotion() { assert!(validate(&[t(&[2],&[1],1),t(&[2],&[1],2),t(&[2],&[1],1)], &[add()],2).is_err()); }
    #[test] fn no_broadcast() { assert!(validate(&[t(&[2],&[1],0),t(&[1],&[1],0),t(&[2],&[1],0)], &[add()],2).is_err()); }
    #[test] fn unknown_operator() { assert!(validate(&[t(&[2],&[1],0),t(&[2],&[1],0)], &[NodeSpec{op:7,a:0,b:0,scalar:0.}],1).is_err()); }
    #[test] fn rms_weight_shape() { assert!(validate(&[t(&[2,3],&[3,1],0),t(&[2],&[1],0),t(&[2,3],&[3,1],0)], &[NodeSpec{op:RMS_NORM,a:0,b:1,scalar:1e-5}],2).is_err()); }
    #[test] fn rms_without_weight() { assert!(validate(&[t(&[2,3],&[3,1],1),t(&[2,3],&[3,1],1)], &[NodeSpec{op:RMS_NORM,a:0,b:NO_WEIGHT,scalar:1e-5}],1).is_ok()); }
    #[test] fn rms_bad_epsilon() { for eps in [0.,-1.,f32::NAN,f32::INFINITY] { assert!(validate(&[t(&[3],&[1],0),t(&[3],&[1],0)], &[NodeSpec{op:RMS_NORM,a:0,b:NO_WEIGHT,scalar:eps}],1).is_err()); } }
    #[test] fn unary_negative_zero_rejected() { assert!(validate(&[t(&[1],&[1],0),t(&[1],&[1],0)], &[NodeSpec{op:SILU,a:0,b:0,scalar:-0.}],1).is_err()); }
    #[test] fn valid_silu_mul_storage_dtypes() { for dt in 0..=2 { assert!(validate(&[t(&[2,3],&[3,1],dt),t(&[2,3],&[3,1],dt),t(&[2,3],&[3,1],dt)], &[NodeSpec{op:SILU_MUL,a:0,b:1,scalar:0.}],2).is_ok()); } }
    #[test] fn silu_mul_noncanonical_scalar() { for scalar in [-0.,1.,f32::NAN,f32::INFINITY] { assert!(validate(&[t(&[2],&[1],0),t(&[2],&[1],0),t(&[2],&[1],0)], &[NodeSpec{op:SILU_MUL,a:0,b:1,scalar}],2).is_err()); } }
    #[test] fn silu_mul_future_input() { assert!(validate(&[t(&[2],&[1],0),t(&[2],&[1],0)], &[NodeSpec{op:SILU_MUL,a:0,b:1,scalar:0.}],1).is_err()); }
    #[test] fn silu_mul_no_promotion() { assert!(validate(&[t(&[2],&[1],1),t(&[2],&[1],2),t(&[2],&[1],1)], &[NodeSpec{op:SILU_MUL,a:0,b:1,scalar:0.}],2).is_err()); }
    #[test] fn silu_mul_no_broadcast() { assert!(validate(&[t(&[2],&[1],0),t(&[1],&[1],0),t(&[2],&[1],0)], &[NodeSpec{op:SILU_MUL,a:0,b:1,scalar:0.}],2).is_err()); }
    #[test] fn silu_mul_code_is_extension_only() { assert_eq!(SILU_MUL,101); assert_ne!(SILU_MUL,SILU); }
    #[test] fn abi_layout() { assert_eq!(std::mem::size_of::<NodeSpec>(),16); assert_eq!(std::mem::align_of::<NodeSpec>(),4); }
}

#[cfg(test)]
mod api3_tests {
    use super::*;
    #[test]
    fn changed_matrix_output_shape_is_validated() {
        fn spec(shape: &'static [usize], strides: &'static [usize]) -> TensorSpec<'static> { TensorSpec{shape,strides,dtype:0} }
        let values=[spec(&[2,3],&[3,1]),spec(&[3,4],&[4,1]),spec(&[2,4],&[4,1])];
        assert!(validate(&values,&[NodeSpec{op:7,a:0,b:1,scalar:0.}],2).is_ok());
    }
    #[test]
    fn reduction_bitmask_and_softmax_axis_checked() {
        let values=[TensorSpec{shape:&[2,3,4],strides:&[12,4,1],dtype:1},
            TensorSpec{shape:&[1,3,1],strides:&[3,1,1],dtype:1}];
        assert!(validate(&values,&[NodeSpec{op:104,a:0,b:0,scalar:5.}],1).is_ok());
        assert!(validate(&values,&[NodeSpec{op:104,a:0,b:0,scalar:2.}],1).is_err());
    }
}
