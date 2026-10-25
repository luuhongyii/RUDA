//! Native SM80 BF16 forward with warp MMA and Flash storage boundaries.
pub mod backward;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{cmma::MmaDefinition, ir::MatrixIdent, prelude::*};
use crate::dense_attention::{flash_exp2,flash_log};

#[ruda(launch)]
pub fn forward<F: Float>(q: &Array<F>, k: &Array<F>, v: &Array<F>,
    out: &mut Array<F>, stats: &mut Array<f32>, heads: u32, kv_heads: u32,
    scale: f32, #[comptime] causal: bool, #[define(F)] _dtype: StorageType) {
    let row_base = RUDA_POS_X as usize * 16;
    let thread = UNIT_POS as usize;
    let lane = thread % 32;
    let warp = thread / 32;
    let head = (row_base / 512) % heads as usize;
    let batch = row_base / (heads as usize * 512);
    let kv_base = (batch * kv_heads as usize + head / (heads / kv_heads) as usize) * 512;
    let def = MmaDefinition::<F,F,f32>::new(16usize,8usize,16usize);
    let mut scores = SharedMemory::<f32>::new(1024usize);
    let mut probability = SharedMemory::<F>::new(1024usize);
    let mut partial = SharedMemory::<f32>::new(64usize);
    let mut factors = SharedMemory::<f32>::new(48usize);
    let mut output_acc = Array::<Vector<f32,Const<2>>>::new(8usize);
    #[unroll]
    for i in 0usize..8 { output_acc[i] = Vector::new(0.0f32); }
    if thread < 16 { factors[thread] = f32::NEG_INFINITY; }
    let soft_row = thread / 4;
    let soft_lane = thread % 4;
    let mut denominator = 0.0f32;
    let scale_log2 = scale * 1.4426950408889634f32;
    let mut blocks = 8usize;
    if comptime!(causal) { blocks = (row_base % 512 / 128 + 1) * 2; }
    sync_ruda();
    for step in 0usize..blocks {
        let block = blocks - 1 - step;
        #[unroll]
        for tile in 0usize..2 {
            let mut a = Array::<Vector<F,Const<2>>>::new(4usize);
            let mut b = Array::<Vector<F,Const<2>>>::new(2usize);
            let mut acc = Array::<Vector<f32,Const<2>>>::new(2usize);
            #[unroll]
            for i in 0usize..2 { acc[i] = Vector::new(0.0f32); }
            #[unroll]
            for chunk in 0usize..8 {
                #[unroll]
                for i in 0usize..4 {
                    let mut value = Vector::<F,Const<2>>::new(F::cast_from(0.0f32));
                    #[unroll]
                    for j in 0usize..2 {
                        let (r,c) = def.position_of_nth(lane as u32,(i*2+j) as u32,MatrixIdent::A);
                        value[j] = q[(row_base+r as usize)*128+chunk*16+c as usize];
                    }
                    a[i] = value;
                }
                #[unroll]
                for i in 0usize..2 {
                    let mut value = Vector::<F,Const<2>>::new(F::cast_from(0.0f32));
                    #[unroll]
                    for j in 0usize..2 {
                        let (r,c) = def.position_of_nth(lane as u32,(i*2+j) as u32,MatrixIdent::B);
                        value[j] = k[(kv_base+block*64+warp*16+tile*8+c as usize)*128+chunk*16+r as usize];
                    }
                    b[i] = value;
                }
                let result = def.execute(&a,&b,&acc);
                #[unroll]
                for i in 0usize..2 { acc[i] = result[i]; }
            }
            #[unroll]
            for i in 0usize..2 {
                #[unroll]
                for j in 0usize..2 {
                    let (r,c) = def.position_of_nth(lane as u32,(i*2+j) as u32,MatrixIdent::Accumulator);
                    let col = warp*16+tile*8+c as usize;
                    let mut score = acc[i][j];
                    if comptime!(causal) { if block*64+col > (row_base+r as usize)%512 { score = f32::NEG_INFINITY; } }
                    scores[r as usize*64+col] = score;
                }
            }
        }
        sync_ruda();
        if thread < 64 {
            let mut maximum = f32::new(f32::NEG_INFINITY);
            #[unroll]
            for i in 0usize..16 {
                let col = (i/2)*8+soft_lane*2+i%2;
                maximum = f32::max(maximum,scores[soft_row*64+col]);
            }
            partial[thread] = maximum;
        }
        sync_ruda();
        if thread < 16 {
            let previous = factors[thread];
            let mut maximum = previous;
            #[unroll]
            for i in 0usize..4 { maximum = f32::max(maximum,partial[thread*4+i]); }
            let mut rescale = 0.0f32;
            if previous != f32::NEG_INFINITY { rescale = flash_exp2((previous-maximum)*scale_log2); }
            factors[thread] = maximum;
            factors[16+thread] = rescale;
        }
        sync_ruda();
        if thread < 64 {
            let maximum = factors[soft_row];
            let mut max_scaled = 0.0f32;
            if maximum != f32::NEG_INFINITY { max_scaled = maximum * scale_log2; }
            #[unroll]
            for i in 0usize..16 {
                let col = (i/2)*8+soft_lane*2+i%2;
                let p = flash_exp2(scores[soft_row*64+col]*scale_log2-max_scaled);
                if i == 0 { denominator = fma(denominator,factors[16+soft_row],p); }
                else { denominator += p; }
                probability[soft_row*64+col] = F::cast_from(p);
            }
        }
        sync_ruda();
        #[unroll]
        for tile in 0usize..4 {
            let mut acc = Array::<Vector<f32,Const<2>>>::new(2usize);
            #[unroll]
            for i in 0usize..2 {
                let mut value = output_acc[tile*2+i];
                #[unroll]
                for j in 0usize..2 {
                    let (r,_c) = def.position_of_nth(lane as u32,(i*2+j) as u32,MatrixIdent::Accumulator);
                    value[j] *= factors[16+r as usize];
                }
                acc[i] = value;
            }
            #[unroll]
            for chunk in 0usize..4 {
                let mut a = Array::<Vector<F,Const<2>>>::new(4usize);
                let mut b = Array::<Vector<F,Const<2>>>::new(2usize);
                #[unroll]
                for i in 0usize..4 {
                    let mut value = Vector::<F,Const<2>>::new(F::cast_from(0.0f32));
                    #[unroll]
                    for j in 0usize..2 {
                        let (r,c) = def.position_of_nth(lane as u32,(i*2+j) as u32,MatrixIdent::A);
                        value[j] = probability[r as usize*64+chunk*16+c as usize];
                    }
                    a[i] = value;
                }
                #[unroll]
                for i in 0usize..2 {
                    let mut value = Vector::<F,Const<2>>::new(F::cast_from(0.0f32));
                    #[unroll]
                    for j in 0usize..2 {
                        let (r,c) = def.position_of_nth(lane as u32,(i*2+j) as u32,MatrixIdent::B);
                        value[j] = v[(kv_base+block*64+chunk*16+r as usize)*128+warp*32+tile*8+c as usize];
                    }
                    b[i] = value;
                }
                let result = def.execute(&a,&b,&acc);
                #[unroll]
                for i in 0usize..2 { acc[i] = result[i]; }
            }
            #[unroll]
            for i in 0usize..2 { output_acc[tile*2+i] = acc[i]; }
        }
        sync_ruda();
    }
    if thread < 64 { partial[thread] = denominator; }
    sync_ruda();
    if thread < 16 {
        let sum = (partial[thread*4]+partial[thread*4+2])+(partial[thread*4+1]+partial[thread*4+3]);
        factors[32+thread] = 1.0f32/sum;
        stats[(row_base+thread)*2] = fma(factors[thread],scale,flash_log(sum));
        stats[(row_base+thread)*2+1] = sum;
    }
    sync_ruda();
    #[unroll]
    for tile in 0usize..4 {
        #[unroll]
        for i in 0usize..2 {
            #[unroll]
            for j in 0usize..2 {
                let (r,c) = def.position_of_nth(lane as u32,(i*2+j) as u32,MatrixIdent::Accumulator);
                let index = (row_base+r as usize)*128+warp*32+tile*8+c as usize;
                let value = F::cast_from(output_acc[tile*2+i][j]*factors[32+r as usize]);
                out[index] = value;
                stats[out.len()/128*2+index] = f32::cast_from(value);
            }
        }
    }
}
