pub use ruda_core::tensor::{QPARAM_ALIGN, params_shape};
use ruda_core::tensor::{QuantLevel, QuantMode, QuantScheme, QuantValue, Shape, Slice, FloatDType, IntDType};
use alloc::vec;

use super::{Calibration, QuantizationParametersPrimitive};
use crate::{Backend, TensorData, TensorMetadata, get_device_settings};

/// Compute the quantization range mapping.
pub fn compute_range<B: Backend>(
    scheme: &QuantScheme,
    tensor: B::FloatTensorPrimitive,
    calibration: &Calibration,
) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive) {
    match calibration {
        Calibration::Percentile { lower, upper } => {
            assert!(lower.is_finite() && upper.is_finite() && 0.0<=*lower && *lower<=*upper && *upper<=1.0);
            let shape=tensor.shape();
            let (blocks, counts)=calibration_blocks::<B>(scheme,tensor,f32::INFINITY);
            let sorted=calibration_sort::<B>(blocks);
            let quantile=|p:f64| {
                let position=B::float_mul_scalar(B::float_sub_scalar(counts.clone(),1f32.into()),p.into());
                let floor=B::float_floor(position.clone());
                let ceil=B::float_ceil(position.clone());
                let fraction=B::float_sub(position,floor.clone());
                let a=B::float_gather(1,sorted.clone(),B::float_into_int(floor,IntDType::I64));
                let b=B::float_gather(1,sorted.clone(),B::float_into_int(ceil,IntDType::I64));
                B::float_reshape(B::float_add(a.clone(),B::float_mul(B::float_sub(b,a),fraction)),params_shape(&shape,scheme.level))
            };
            (quantile(*lower),quantile(*upper))
        },
        Calibration::Mse { steps } => {
            assert!(*steps>0,"MSE calibration requires a nonzero search size");
            assert!(matches!(scheme.value,QuantValue::Q2F|QuantValue::Q2S|QuantValue::Q4F|QuantValue::Q4S|QuantValue::Q8F|QuantValue::Q8S),
                "generic MSE calibration uses integer quantization levels");
            let shape=tensor.shape();
            let (blocks,_)=calibration_blocks::<B>(scheme,tensor,0.0);
            let bool_dtype=get_device_settings::<B>(&B::float_device(&blocks)).bool_dtype;
            let absolute=B::float_max_dim(B::float_abs(blocks.clone()),1);
            let (a,b)=scheme.value.range();
            let mut best_range=absolute.clone();
            let mut best_error:Option<B::FloatTensorPrimitive>=None;
            for step in 0..*steps {
                let range=B::float_mul_scalar(absolute.clone(),(1f64-step as f64/(*steps as f64)).into());
                let scale=B::float_div_scalar(B::float_mul_scalar(range.clone(),2f32.into()),(b-a).into());
                let zero=B::float_equal_elem(scale.clone(),0f32.into(),bool_dtype);
                let safe=B::float_mask_fill(scale.clone(),zero,1f32.into());
                let code=B::float_clamp(B::float_round(B::float_div(blocks.clone(),safe)),a.into(),b.into());
                let delta=B::float_sub(B::float_mul(code,scale),blocks.clone());
                let error=B::float_sum_dim(B::float_mul(delta.clone(),delta),1);
                if let Some(old)=best_error.take() {
                    let better=B::float_lower(error.clone(),old.clone(),bool_dtype);
                    best_range=B::float_mask_where(best_range,better.clone(),range);
                    best_error=Some(B::float_mask_where(old,better,error));
                } else {best_error=Some(error);}
            }
            let high=B::float_reshape(best_range,params_shape(&shape,scheme.level));
            (B::float_mul_scalar(high.clone(),(-1f32).into()),high)
        },
        Calibration::MinMax => match scheme.level {
            QuantLevel::Tensor => (B::float_min(tensor.clone()), B::float_max(tensor)),
            QuantLevel::Block(block_size) => {
                let shape = tensor.shape();
                let block_dims = block_size.to_dim_vec(shape.rank());
                assert!(!block_dims.contains(&0), "Quantization block dimensions must be nonzero");
                let params_shape = params_shape(&shape, scheme.level);
                if shape.num_elements() == 0 {
                    let device = B::float_device(&tensor);
                    let dtype = tensor.dtype().into();
                    return (
                        B::float_empty(params_shape.clone(), &device, dtype),
                        B::float_empty(params_shape, &device, dtype),
                    );
                }

                let mut contiguous_blocks = true;
                let mut inner_axes_complete = true;
                for (&dim, &block) in shape.iter().zip(&block_dims).rev() {
                    let block = block as usize;
                    contiguous_blocks &= dim.is_multiple_of(block)
                        && (block == 1 || inner_axes_complete);
                    inner_axes_complete &= dim == block;
                }
                if contiguous_blocks {
                    let num_blocks = params_shape.num_elements();
                    let block_elems = shape.num_elements() / num_blocks;
                    let blocks = B::float_reshape(tensor, Shape::new([num_blocks, block_elems]));
                    return (
                        B::float_reshape(B::float_min_dim(blocks.clone(), 1), params_shape.clone()),
                        B::float_reshape(B::float_max_dim(blocks, 1), params_shape),
                    );
                }

                let mut min = tensor.clone();
                let mut max = tensor;
                for (axis, &block) in block_dims.iter().enumerate().rev() {
                    if block > 1 {
                        min = reduce_block_axis::<B>(min, axis, block as usize, B::float_min_dim);
                        max = reduce_block_axis::<B>(max, axis, block as usize, B::float_max_dim);
                    }
                }
                (min, max)
            }
        },
    }
}

/// Group arbitrary multidimensional blocks, including partial edge blocks.
/// Only geometry counts are constructed on host; sample values remain on device.
fn calibration_blocks<B:Backend>(scheme:&QuantScheme,tensor:B::FloatTensorPrimitive,padding:f32)
    ->(B::FloatTensorPrimitive,B::FloatTensorPrimitive) {
    let shape=tensor.shape();assert!(shape.num_elements()>0,"calibration requires nonempty input");
    let device=B::float_device(&tensor);
    let tensor=B::float_cast(tensor,FloatDType::F32);
    if matches!(scheme.level,QuantLevel::Tensor) {
        return (B::float_reshape(tensor,Shape::new([1,shape.num_elements()])),
            B::float_full(Shape::new([1,1]),(shape.num_elements() as f32).into(),&device,FloatDType::F32));
    }
    let QuantLevel::Block(size)=scheme.level else {unreachable!()};
    let blocks=size.to_dim_vec(shape.rank());assert!(!blocks.contains(&0));
    let params=params_shape(&shape,scheme.level);
    let mut padded=tensor;
    for (axis,&block) in blocks.iter().enumerate() {
        let mut tail=padded.shape();
        tail[axis]=(block as usize-shape[axis]%(block as usize))%(block as usize);
        if tail[axis]>0 {
            let fill=B::float_full(tail,padding.into(),&device,FloatDType::F32);
            padded=B::float_cat(vec![padded,fill],axis);
        }
    }
    let mut interleaved=alloc::vec::Vec::new();
    for (&n,&b) in params.iter().zip(&blocks) {interleaved.push(n);interleaved.push(b as usize);}
    let axes:alloc::vec::Vec<_>=(0..shape.rank()).map(|i|2*i).chain((0..shape.rank()).map(|i|2*i+1)).collect();
    let elements:usize=blocks.iter().map(|&n|n as usize).product();
    let grouped=B::float_reshape(B::float_permute(B::float_reshape(padded,Shape::from(interleaved)),&axes),
        Shape::new([params.num_elements(),elements]));
    let counts:alloc::vec::Vec<f32>=(0..params.num_elements()).map(|mut index| {
        let mut count=1usize;
        for axis in (0..shape.rank()).rev() {
            let pos=index%params[axis];index/=params[axis];
            count*=usize::min(blocks[axis] as usize,shape[axis]-pos*blocks[axis] as usize);
        }
        count as f32
    }).collect();
    (grouped,B::float_from_data(TensorData::new(counts,Shape::new([params.num_elements(),1])),&device))
}

// A device-composed sorting network avoids the generic backend sort default,
// which is allowed to read values back to the host. Only permutation metadata
// are constructed on CPU. Padding is excluded by the per-block valid counts.
fn calibration_sort<B:Backend>(mut values:B::FloatTensorPrimitive)->B::FloatTensorPrimitive {
    let shape=values.shape();let rows=shape[0];let width=shape[1];
    let padded=width.checked_next_power_of_two().expect("calibration sort size overflow");
    let device=B::float_device(&values);
    let bool_dtype=get_device_settings::<B>(&device).bool_dtype;
    if padded>width {
        let tail=B::float_full(Shape::new([rows,padded-width]),f32::INFINITY.into(),&device,FloatDType::F32);
        values=B::float_cat(vec![values,tail],1);
    }
    let mut span=2usize;
    while span<=padded {
        let mut distance=span/2;
        while distance>0 {
            let permutation:alloc::vec::Vec<i64>=(0..padded).map(|i|(i^distance) as i64).collect();
            let direction:alloc::vec::Vec<f32>=(0..padded).map(|i|if (i&span==0)==(i&distance==0) {1.0} else {-1.0}).collect();
            let index=B::int_expand(B::int_from_data(TensorData::new(permutation,Shape::new([1,padded])),&device),Shape::new([rows,padded]));
            let direction=B::float_from_data(TensorData::new(direction,Shape::new([1,padded])),&device);
            let partner=B::float_gather(1,values.clone(),index);
            let replace=B::float_greater(B::float_mul(values.clone(),direction.clone()),B::float_mul(partner.clone(),direction),bool_dtype);
            values=B::float_mask_where(values,replace,partner);
            distance/=2;
        }
        if span==padded {break;}
        span*=2;
    }
    B::float_slice(values,&[Slice::from(..),Slice::from(0..width)])
}

/// Explicit affine parameters, independent of the symmetric QFloat byte ABI.
pub struct AffineQuantizationParametersPrimitive<B:Backend> {
    pub scales:B::FloatTensorPrimitive,
    pub zero_points:B::FloatTensorPrimitive,
    pub offsets:B::FloatTensorPrimitive,
}

/// Unsigned integer affine parameters, including zero exactly in the range.
pub fn compute_affine_q_params<B:Backend>(min:B::FloatTensorPrimitive,max:B::FloatTensorPrimitive,bits:u8)
    ->AffineQuantizationParametersPrimitive<B> {
    assert!((2..=8).contains(&bits));
    let min=B::float_clamp_max(B::float_cast(min,FloatDType::F32),0f32.into());
    let max=B::float_clamp_min(B::float_cast(max,FloatDType::F32),0f32.into());
    let bool_dtype=get_device_settings::<B>(&B::float_device(&min)).bool_dtype;
    let maximum=((1u32<<bits)-1) as f32;
    let scales=B::float_div_scalar(B::float_sub(max,min.clone()),maximum.into());
    let mask=B::float_equal_elem(scales.clone(),0f32.into(),bool_dtype);
    let safe=B::float_mask_fill(scales.clone(),mask,1f32.into());
    let zero_points=B::float_clamp(B::float_round(B::float_div(B::float_mul_scalar(min,(-1f32).into()),safe)),0f32.into(),maximum.into());
    let offsets=B::float_mul_scalar(B::float_mul(zero_points.clone(),scales.clone()),(-1f32).into());
    AffineQuantizationParametersPrimitive {scales,zero_points,offsets}
}

fn reduce_block_axis<B: Backend>(
    tensor: B::FloatTensorPrimitive,
    axis: usize,
    block: usize,
    reduce: fn(B::FloatTensorPrimitive, usize) -> B::FloatTensorPrimitive,
) -> B::FloatTensorPrimitive {
    let shape = tensor.shape();
    if shape[axis] <= block {
        return reduce(tensor, axis);
    }
    let full_blocks = shape[axis] / block;
    let full_len = full_blocks * block;
    let has_tail = full_len != shape[axis];
    let mut slices = vec![Slice::from(..); shape.rank()];
    let head = if has_tail {
        slices[axis] = Slice::from(0..full_len);
        B::float_slice(tensor.clone(), &slices)
    } else {
        tensor.clone()
    };
    let mut grouped_shape = shape.clone();
    grouped_shape[axis] = full_blocks;
    grouped_shape.insert(axis + 1, block);
    let head = reduce(B::float_reshape(head, grouped_shape), axis + 1);
    let mut output_shape = shape;
    output_shape[axis] = full_blocks;
    let head = B::float_reshape(head, output_shape);
    if has_tail {
        slices[axis] = Slice::from(full_len..);
        let tail = reduce(B::float_slice(tensor, &slices), axis);
        B::float_cat(vec![head, tail], axis)
    } else {
        head
    }
}

/// Compute the quantization parameters.
pub fn compute_q_params<B: Backend>(
    scheme: &QuantScheme,
    min: B::FloatTensorPrimitive,
    max: B::FloatTensorPrimitive,
) -> QuantizationParametersPrimitive<B> {
    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            ..
        } => {
            let bool_dtype = get_device_settings::<B>(&B::float_device(&min)).bool_dtype;
            // Quantized range `[a, b]`
            let (a, b) = scheme.value.range();

            // Compute scale to convert an input value in range `[-alpha, alpha]`
            let min_abs = B::float_abs(min);
            let max_abs = B::float_abs(max);

            // `min_abs.max_pair(max_abs)`
            let mask = B::float_lower(min_abs.clone(), max_abs.clone(), bool_dtype);
            let values_range =
                B::float_mul_scalar(B::float_mask_where(min_abs, mask, max_abs), 2f32.into());

            QuantizationParametersPrimitive {
                scales: B::float_div_scalar(values_range, (b - a).into()),
            }
        }
    }
}
