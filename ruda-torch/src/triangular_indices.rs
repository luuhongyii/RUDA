use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda(launch)]
fn generate<I: Int + RudaElement>(
    out: &mut Tensor<I>, columns: i64, row_start: i64, first: i64,
    ramp_rows: i64, boundary: i64, #[comptime] upper: bool,
) {
    let p = ABSOLUTE_POS as usize;
    let count = out.shape(1);
    if p < count {
        let index = i64::cast_from(p);
        let mut row = 0i64;
        let mut column = 0i64;
        let mut ramp = index < boundary;
        let mut within = index;
        if comptime!(upper) {
            ramp = index >= boundary;
            within = index - boundary;
        }
        if ramp {
            let mut low = 0i64;
            let mut high = ramp_rows;
            while low < high {
                let middle = low + (high - low) / 2i64;
                let n = middle + 1i64;
                let mut prefix = n * (2i64 * first + middle) / 2i64;
                if comptime!(upper) { prefix = n * (2i64 * first - middle) / 2i64; }
                if prefix <= within { low = n; } else { high = middle; }
            }
            let mut before = low * (2i64 * first + low - 1i64) / 2i64;
            if comptime!(upper) { before = low * (2i64 * first - low + 1i64) / 2i64; }
            row = row_start + low;
            column = within - before;
            if comptime!(upper) { column += columns - first + low; }
        } else if comptime!(upper) {
            row = index / columns;
            column = index % columns;
        } else {
            let surplus = index - boundary;
            row = row_start + ramp_rows + surplus / columns;
            column = surplus % columns;
        }
        out[address(out, p)] = I::cast_from(row);
        out[address(out, p + count)] = I::cast_from(column);
    }
}

pub(super) fn dispatch(op: u32, xs: &[View], ints: &[i64]) {
    assert!(matches!(op, 411 | 412));
    assert_eq!(xs.len(), 1);
    assert_eq!(ints.len(), 5);
    assert!(ints.iter().all(|&n| n >= 0));
    let out = &xs[0];
    assert_eq!(out.shape.len(), 2);
    assert_eq!(out.shape[0], 2);
    assert!(matches!(out.dtype, 4 | 5));
    if out.len == 0 { return; }
    let count = out.shape[1];
    assert!(ints[0] > 0 && ints[2] > 0 && ints[2] <= ints[0]);
    assert!(ints[2] <= count as i64 && ints[3] <= count as i64 && ints[4] <= count as i64);
    let c = client();
    let grid = RudaCount::Static(u32::try_from(count.div_ceil(256)).expect("triangular indices launch overflow"), 1, 1);
    let block = RudaDim::new_1d(256);
    macro_rules! run { ($i:ty) => { unsafe {
        generate::launch::<$i, CudaRuntime>(&c, grid, block, out.arg(),
            ints[0], ints[1], ints[2], ints[3], ints[4], op == 412);
    }}; }
    match out.dtype { 4 => run!(i64), 5 => run!(i32), _ => unreachable!() }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    finish_dispatch(&c);
}
