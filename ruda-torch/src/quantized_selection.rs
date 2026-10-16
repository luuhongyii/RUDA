use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn compare<A: Numeric + RudaElement, B: Numeric + RudaElement>(
    a: &Tensor<A>, b: &Tensor<B>, output: &mut Tensor<u8>,
    a_scale: f64, b_scale: f64, a_zero: i64, b_zero: i64, #[comptime] operation: u32,
) {
    let p = ABSOLUTE_POS as usize;
    if p < output.len() {
        let av = f32::cast_from(a[address(a, p)]) - f32::cast_from(a_zero);
        let bv = f32::cast_from(b[address(b, p)]) - f32::cast_from(b_zero);
        let x = f32::cast_from(f64::cast_from(av) * a_scale);
        let y = f32::cast_from(f64::cast_from(bv) * b_scale);
        let mut value = x == y;
        if comptime!(operation == 1) { value = x != y; }
        else if comptime!(operation == 2) { value = x > y; }
        else if comptime!(operation == 3) { value = x >= y; }
        else if comptime!(operation == 4) { value = x < y; }
        else if comptime!(operation == 5) { value = x <= y; }
        output[address(output, p)] = u8::cast_from(value);
    }
}

#[ruda(launch)]
fn equal_blocks<E: Numeric + RudaElement>(
    a: &Tensor<E>, b: &Tensor<E>, flags: &mut Tensor<u8>, #[comptime] boolean: bool,
) {
    let p = ABSOLUTE_POS as usize; let lane = UNIT_POS as usize;
    let mut equal = 1u32;
    if p < a.len() {
        let x = a[address(a, p)]; let y = b[address(b, p)];
        if comptime!(boolean) { equal = u32::cast_from((x != E::cast_from(0u32)) == (y != E::cast_from(0u32))); }
        else { equal = u32::cast_from(x == y); }
    }
    let mut shared = SharedMemory::<u32>::new(256usize);
    shared[lane] = equal; sync_ruda();
    let mut step = 128usize;
    while step > 0usize {
        if lane < step {
            let combined = shared[lane] & shared[lane + step];
            shared[lane] = combined;
        }
        sync_ruda(); step /= 2usize;
    }
    if lane == 0usize { flags[RUDA_POS_X as usize] = u8::cast_from(shared[0]); }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64], values: &[f64]) {
    assert!(op == 572 || op == 573); assert_eq!(xs.len(), 3);
    let a = &xs[0]; let b = &xs[1]; let output = &xs[2];
    assert_eq!(a.shape, b.shape); assert_eq!(output.dtype, 3);
    let c = client(); let block = RudaDim::new_1d(256);
    if op == 572 {
        assert_eq!(ints.len(), 3); assert_eq!(values.len(), 2); assert!((0..=5).contains(&ints[0]));
        assert_eq!(a.shape, output.shape);
        for (view, &zero) in [a,b].iter().zip(&ints[1..]) {
            let (minimum, maximum): (i64, i64) = match view.dtype {
                5 => (i32::MIN as i64, i32::MAX as i64), 7 => (-128,127), 8 => (0,255),
                _ => panic!("quantized comparison requires int8, uint8 or int32 payload"),
            };
            assert!((minimum..=maximum).contains(&zero));
        }
        if output.len == 0 { return; }
        let grid = RudaCount::Static(u32::try_from(output.len.div_ceil(256)).expect("comparison launch overflow"),1,1);
        macro_rules! run { ($a:ty,$b:ty) => { unsafe {
            compare::launch::<$a,$b,CudaRuntime>(&c,grid,block,a.arg(),b.arg(),output.arg(),
                values[0],values[1],ints[1],ints[2],ints[0] as u32);
        } }; }
        macro_rules! right { ($a:ty) => { match b.dtype {5=>run!($a,i32),7=>run!($a,i8),8=>run!($a,u8),_=>unreachable!()} }; }
        match a.dtype {5=>right!(i32),7=>right!(i8),8=>right!(u8),_=>unreachable!()}
    } else {
        assert!(ints.is_empty() && values.is_empty()); assert_eq!(a.dtype, b.dtype); assert!(a.dtype <= 9);
        assert_eq!(output.shape, vec![a.len.div_ceil(256)]); assert_eq!(output.strides, vec![1]);
        if a.len == 0 { return; }
        let grid = RudaCount::Static(u32::try_from(output.len).expect("equality launch overflow"),1,1);
        macro_rules! run { ($e:ty) => { unsafe {
            equal_blocks::launch::<$e,CudaRuntime>(&c,grid,block,a.arg(),b.arg(),output.arg(),a.dtype == 3);
        } }; }
        match a.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),3|8=>run!(u8),4=>run!(i64),5=>run!(i32),
            6=>run!(i16),7=>run!(i8),9=>run!(f64),_=>unreachable!()}
    }
}
