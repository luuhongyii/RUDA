use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::{Tensor, fma};

#[ruda]
fn load_float(input: &Tensor<u8>, offset: usize) -> f32 {
    let bits = u32::cast_from(input[offset]) | (u32::cast_from(input[offset + 1usize]) << 8u32)
        | (u32::cast_from(input[offset + 2usize]) << 16u32) | (u32::cast_from(input[offset + 3usize]) << 24u32);
    f32::reinterpret(bits)
}

#[ruda]
fn store_float(output: &mut Tensor<u8>, offset: usize, value: f32) {
    let bits = u32::reinterpret(value);
    #[unroll]
    for byte in 0..4usize { output[offset + byte] = u8::cast_from((bits >> u32::cast_from(byte * 8usize)) & 255u32); }
}

#[ruda(launch)]
fn pack_float<F: Float + RudaElement>(
    input: &Tensor<F>, minmax: &Tensor<F>, output: &mut Tensor<u8>, #[comptime] supplied: bool,
) {
    let row = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if row < input.shape(0) {
        let lane = UNIT_POS as usize; let width = input.shape(1);
        let mut minima = SharedMemory::<f32>::new(8usize);
        let mut maxima = SharedMemory::<f32>::new(8usize);
        let mut params = SharedMemory::<f32>::new(3usize);
        if comptime!(!supplied) {
            if lane < 8usize {
                let mut minimum = f32::reinterpret(0x7f7fffffu32); let mut maximum = -minimum;
                let mut col = lane;
                while col < width / 8usize * 8usize {
                    let x = f32::cast_from(input[address(input, row * width + col)]);
                    if !(minimum < x) { minimum = x; }
                    if !(maximum > x) { maximum = x; }
                    col += 8usize;
                }
                minima[lane] = minimum; maxima[lane] = maximum;
            }
        }
        sync_ruda();
        if lane == 0usize {
            let mut minimum = f32::reinterpret(0x7f7fffffu32); let mut maximum = -minimum;
            if comptime!(supplied) {
                minimum = f32::cast_from(minmax[address(minmax, row * 2usize)]);
                maximum = f32::cast_from(minmax[address(minmax, row * 2usize + 1usize)]);
            } else {
                for part in 0..8usize {
                    if minima[part] < minimum { minimum = minima[part]; }
                    if maxima[part] > maximum { maximum = maxima[part]; }
                }
                for col in width / 8usize * 8usize..width {
                    let x = f32::cast_from(input[address(input, row * width + col)]);
                    if x < minimum { minimum = x; }
                    if x > maximum { maximum = x; }
                }
            }
            let range = maximum - minimum;
            params[0] = range / 255.0f32; params[1] = minimum; params[2] = 255.0f32 / (range + 1.0e-8f32);
            store_float(output, row * (width + 8usize) + width, params[0]);
            store_float(output, row * (width + 8usize) + width + 4usize, minimum);
        }
        sync_ruda();
        let mut col = lane;
        while col < width {
            let x = f32::cast_from(input[address(input, row * width + col)]);
            let rounded = ((x - params[1]) * params[2]).round();
            let mut q = i64::cast_from(rounded);
            if col < width / 8usize * 8usize {
                if !(rounded >= -2147483648.0f32 && rounded < 2147483648.0f32) { q = -2147483648i64; }
            }
            if col < width / 32usize * 32usize {
                if q < 0i64 { q = 0i64; }
                if q > 255i64 { q = 255i64; }
            }
            output[row * (width + 8usize) + col] = u8::cast_from(q & 255i64);
            col += 256usize;
        }
    }
}

#[ruda(launch)]
fn pack_quantized(input: &Tensor<u8>, scales: &Tensor<f32>, zeros: &Tensor<f32>, output: &mut Tensor<u8>) {
    let row = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if row < input.shape(0) {
        let lane = UNIT_POS as usize; let width = input.shape(1);
        if lane == 0usize {
            let scale = scales[address(scales, row)]; let bias = zeros[address(zeros, row)] * scale * -1.0f32;
            store_float(output, row * (width + 8usize) + width, scale);
            store_float(output, row * (width + 8usize) + width + 4usize, bias);
        }
        let mut col = lane;
        while col < width {
            output[row * (width + 8usize) + col] = input[address(input, row * width + col)];
            col += 256usize;
        }
    }
}

#[ruda(launch)]
fn unpack(input: &Tensor<u8>, output: &mut Tensor<f32>) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let width = output.shape(1); let col = p % width; let base = p / width * (width + 8usize);
        let scale = load_float(input, base + width); let bias = load_float(input, base + width + 4usize);
        let q = f32::cast_from(input[base + col]); let mut value = q * scale + bias;
        if col < width / 8usize * 8usize { value = fma(q, scale, bias); }
        output[p] = value;
    }
}

#[ruda]
fn sparse_mapping(mapping: &Tensor<i32>, #[comptime] pruned: bool) -> bool {
    let mut sparse = false;
    if comptime!(pruned) {
        sparse = true;
        if mapping.len() == 1usize { if mapping[0] == 0i32 { sparse = false; } }
    }
    sparse
}

#[ruda]
fn bag_offset<O: Int + RudaElement>(
    offsets: &Tensor<O>, bag: usize, count: usize, length: usize, fixed: usize,
    #[comptime] kind: u32, #[comptime] include_last: bool,
) -> i64 {
    let mut result = i64::cast_from(bag);
    if comptime!(kind == 1) {
        result = i64::cast_from(length);
        if bag < count || comptime!(include_last) { result = i64::cast_from(offsets[bag]); }
    } else if comptime!(kind == 2) { result = i64::cast_from(bag * fixed); }
    result
}

#[ruda(launch)]
fn check_lookup<I: Int + RudaElement, O: Int + RudaElement>(
    indices: &Tensor<I>, offsets: &Tensor<O>, mapping: &Tensor<i32>, flags: &mut Tensor<u8>,
    rows: usize, bags: usize, fixed: usize, work: usize,
    #[comptime] kind: u32, #[comptime] include_last: bool, #[comptime] pruned: bool,
) {
    let block = RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize;
    if block < flags.len() {
        let lane = UNIT_POS as usize; let p = block * 256usize + lane;
        let sparse = sparse_mapping(mapping, pruned); let length = i64::cast_from(indices.len());
        let mut valid = true;
        if p < work {
            if bags == 0usize {
                if sparse { valid = length == 0i64; }
            } else {
                let first = bag_offset(offsets, 0usize, bags, indices.len(), fixed, kind, include_last);
                let last = bag_offset(offsets, bags, bags, indices.len(), fixed, kind, include_last);
                let mut start = first; let mut end = last;
                if sparse { start = 0i64; end = last - first; }
                valid = start >= 0i64 && end >= start && end <= length && last >= first;
                if sparse { valid = valid && end == length; }
                if comptime!(kind == 1) {
                    if p < offsets.len() {
                        let offset = i64::cast_from(offsets[p]); let mut relative = offset;
                        if sparse { relative -= first; }
                        valid = valid && relative >= 0i64 && relative <= length;
                        if p > 0usize { valid = valid && offset >= i64::cast_from(offsets[p - 1usize]); }
                    }
                }
                if valid && p < indices.len() && i64::cast_from(p) >= start && i64::cast_from(p) < end {
                    let index = i64::cast_from(indices[p]);
                    if sparse {
                        valid = index >= 0i64 && index < i64::cast_from(mapping.len());
                        if valid {
                            let mapped = i64::cast_from(mapping[usize::cast_from(index)]);
                            valid = mapped == -1i64 || (mapped >= 0i64 && mapped < i64::cast_from(rows));
                        }
                    } else { valid = index >= 0i64 && index < i64::cast_from(rows); }
                }
            }
        }
        let mut shared = SharedMemory::<u32>::new(256usize);
        shared[lane] = u32::cast_from(valid); sync_ruda();
        let mut step = 128usize;
        while step > 0usize {
            if lane < step { let both = shared[lane] & shared[lane + step]; shared[lane] = both; }
            sync_ruda(); step /= 2usize;
        }
        if lane == 0usize { flags[block] = u8::cast_from(shared[0]); }
    }
}

#[ruda(launch)]
fn lookup<I: Int + RudaElement, O: Int + RudaElement>(
    packed: &Tensor<u8>, indices: &Tensor<I>, offsets: &Tensor<O>,
    weights: &Tensor<f32>, mapping: &Tensor<i32>, output: &mut Tensor<f32>, fixed: usize,
    #[comptime] kind: u32, #[comptime] include_last: bool, #[comptime] pruned: bool, #[comptime] weighted: bool,
) {
    let p = (RUDA_POS_X as usize + RUDA_POS_Y as usize * RUDA_COUNT_X as usize) * 256usize + UNIT_POS as usize;
    if p < output.len() {
        let width = output.shape(1); let bag = p / width; let col = p % width; let bags = output.shape(0);
        let sparse = sparse_mapping(mapping, pruned);
        let mut start = bag_offset(offsets, bag, bags, indices.len(), fixed, kind, include_last);
        let mut end = bag_offset(offsets, bag + 1usize, bags, indices.len(), fixed, kind, include_last);
        if sparse {
            let first = bag_offset(offsets, 0usize, bags, indices.len(), fixed, kind, include_last);
            start -= first; end -= first;
        }
        let mut total = 0.0f32; let mut sample = usize::cast_from(start);
        while sample < usize::cast_from(end) {
            let mut index = i64::cast_from(indices[sample]);
            if sparse { index = i64::cast_from(mapping[usize::cast_from(index)]); }
            if index >= 0i64 {
                let base = usize::cast_from(index) * (width + 8usize);
                let mut scale = load_float(packed, base + width); let mut bias = load_float(packed, base + width + 4usize);
                if comptime!(weighted) { let weight = weights[sample]; scale *= weight; bias *= weight; }
                total = fma(f32::cast_from(packed[base + col]), scale, total + bias);
            }
            sample += 1usize;
        }
        output[p] = total;
    }
}

fn block_grid(blocks: usize) -> RudaCount {
    let x = blocks.min(131072);
    RudaCount::Static(x as u32, blocks.div_ceil(x) as u32, 1)
}

fn contiguous(view: &View) -> bool {
    if view.len == 0 { return true; }
    let mut stride = 1usize;
    for (&size, &actual) in view.shape.iter().zip(&view.strides).rev() {
        if size > 1 && stride != actual { return false; }
        stride *= size;
    }
    true
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!((579..=583).contains(&op)); assert!(values.is_empty());
    let c = client(); let block = RudaDim::new_1d(256);
    if op == 579 || op == 580 {
        assert_eq!(xs.len(), if op == 579 { 3 } else { 4 });
        assert_eq!(ints.len(), if op == 579 { 1 } else { 0 });
        let input = &xs[0]; let output = xs.last().unwrap();
        assert_eq!(input.shape.len(), 2); assert_eq!(output.dtype, 8); assert!(contiguous(output));
        let rows = input.shape[0]; let width = input.shape[1];
        assert_eq!(output.shape, vec![rows,width.checked_add(8).expect("embedding packed width overflow")]);
        if op == 579 {
            assert!((0..=1).contains(&ints[0])); assert!(matches!(input.dtype, 0 | 1));
            assert_eq!(xs[1].dtype, input.dtype);
            if ints[0] != 0 { assert!(xs[1].len >= rows * 2); }
        } else {
            assert_eq!(input.dtype, 8);
            for parameter in &xs[1..3] { assert_eq!(parameter.dtype, 0); assert_eq!(parameter.len, rows); }
        }
        if rows == 0 { return; }
        if op == 579 {
            macro_rules! run { ($f:ty) => { unsafe {
                pack_float::launch::<$f,CudaRuntime>(&c,block_grid(rows),block,input.arg(),xs[1].arg(),output.arg(),ints[0] != 0);
            } }; }
            if input.dtype == 0 { run!(f32); } else { run!(f16); }
        } else { unsafe {
            pack_quantized::launch::<CudaRuntime>(&c,block_grid(rows),block,input.arg(),xs[1].arg(),xs[2].arg(),output.arg());
        } }
    } else if op == 581 {
        assert_eq!(xs.len(), 2); assert!(ints.is_empty());
        let (input, output) = (&xs[0], &xs[1]); assert_eq!(input.dtype, 8); assert_eq!(output.dtype, 0);
        assert_eq!(input.shape.len(), 2); assert_eq!(output.shape.len(), 2);
        assert_eq!(input.shape, vec![output.shape[0],output.shape[1] + 8]);
        assert!(contiguous(input) && contiguous(output));
        if output.len == 0 { return; }
        unsafe { unpack::launch::<CudaRuntime>(&c,block_grid(output.len.div_ceil(256)),block,input.arg(),output.arg()); }
    } else {
        assert_eq!(xs.len(), 6); assert_eq!(ints.len(), 6);
        let (packed, indices, offsets, weights, mapping, output) = (&xs[0],&xs[1],&xs[2],&xs[3],&xs[4],&xs[5]);
        assert_eq!(packed.dtype, 8); assert_eq!(packed.shape.len(), 2); assert!(packed.shape[1] >= 8);
        assert!(matches!(indices.dtype, 4 | 5) && matches!(offsets.dtype, 4 | 5));
        assert_eq!(weights.dtype, 0); assert_eq!(mapping.dtype, 5);
        for view in xs { assert!(contiguous(view)); }
        for view in &xs[1..5] { assert_eq!(view.shape.len(), 1); }
        assert!((0..=2).contains(&ints[0])); assert!(ints[1] >= 0 && ints[2] >= 0);
        for &flag in &ints[3..6] { assert!((0..=1).contains(&flag)); }
        let kind = ints[0] as u32; let bags = ints[1] as usize; let fixed = ints[2] as usize;
        let include_last = ints[3] != 0; let pruned = ints[4] != 0; let weighted = ints[5] != 0;
        if weighted { assert_eq!(weights.len, indices.len); } else { assert_eq!(weights.len, 0); }
        if !pruned { assert_eq!(mapping.len, 0); }
        if kind == 1 { assert_eq!(offsets.len, bags + usize::from(include_last)); }
        else {
            assert_eq!(offsets.len, 0);
            if kind == 0 { assert_eq!(bags, indices.len); assert!(!include_last); }
            else { assert!(fixed > 0); assert_eq!((bags + usize::from(include_last)).checked_mul(fixed), Some(indices.len)); }
        }
        let work = indices.len.max(offsets.len).max(1); let check_blocks = work.div_ceil(256);
        if op == 582 { assert_eq!(output.dtype, 3); assert_eq!(output.shape, vec![check_blocks]); }
        else {
            assert_eq!(output.dtype, 0); assert_eq!(output.shape, vec![bags,packed.shape[1] - 8]);
            if output.len == 0 { return; }
        }
        macro_rules! run { ($i:ty,$o:ty) => { unsafe {
            if op == 582 {
                check_lookup::launch::<$i,$o,CudaRuntime>(&c,block_grid(check_blocks),block,
                    indices.arg(),offsets.arg(),mapping.arg(),output.arg(),packed.shape[0],bags,fixed,work,kind,include_last,pruned);
            } else {
                lookup::launch::<$i,$o,CudaRuntime>(&c,block_grid(output.len.div_ceil(256)),block,
                    packed.arg(),indices.arg(),offsets.arg(),weights.arg(),mapping.arg(),output.arg(),fixed,kind,include_last,pruned,weighted);
            }
        } }; }
        match (indices.dtype,offsets.dtype) { (4,4)=>run!(i64,i64),(4,5)=>run!(i64,i32),(5,4)=>run!(i32,i64),(5,5)=>run!(i32,i32),_=>unreachable!() }
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
}
