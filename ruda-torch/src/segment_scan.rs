use super::*;

#[ruda(launch)]
fn scan(input:&Tensor<f32>,out:&mut Tensor<f32>,#[comptime] width:usize,#[comptime] exponential:bool) {
    let column=ABSOLUTE_POS as usize;
    if column<input.len() {
        let batch=column/width;let start=column%width;
        let mut total=0.0f32;
        for row in 0usize..width {
            if row>start {total+=input[batch*width+row];}
            let mut value=total;
            if row<start {value=f32::new(f32::NEG_INFINITY);}
            if comptime!(exponential) {value=value.exp();}
            out[(batch*width+row)*width+start]=value;
        }
    }
}

#[ruda(launch)]
fn warp_scan(input:&Tensor<f32>,out:&mut Tensor<f32>,#[comptime] width:usize,#[comptime] exponential:bool) {
    let warp=ABSOLUTE_POS as usize/32;let lane=UNIT_POS as usize%32;
    let tiles=comptime!(width.div_ceil(32));let batch=warp/tiles;let column=warp%tiles*32+lane;
    if batch<input.len()/width {
        let mut total=0.0f32;
        for tile in 0usize..tiles {
            let source=tile*32+lane;let mut loaded=0.0f32;
            if source<width {loaded=input[batch*width+source];}
            #[unroll] for j in 0usize..32 {
                let row=tile*32+j;let incoming=plane_shuffle(loaded,j as u32);
                if row<width && column<width {
                    let mut value=f32::new(f32::NEG_INFINITY);
                    if comptime!(exponential) {value=0.0;}
                    if row==column {
                        value=0.0;if comptime!(exponential) {value=1.0;}
                    } else if row>column {
                        total+=incoming;value=total;
                        if comptime!(exponential) {value=value.exp();}
                    }
                    out[(batch*width+row)*width+column]=value;
                }
            }
        }
    }
}

pub(super) fn prepare(client:&ComputeClient<CudaRuntime>,input:&View,out:&View,exponential:bool)->PreparedKernel<CudaRuntime> {
    let width=*input.shape.last().unwrap();
    let h=&client.properties().hardware;
    let warps=input.len/width*width.div_ceil(32);
    if h.plane_size_min==32 && h.plane_size_max==32 && h.max_units_per_ruda>=128 && h.max_ruda_dim.0>=128
        && warps.div_ceil(4)<=h.max_ruda_count.0 as usize {
        return unsafe {warp_scan::prepare::<CudaRuntime>(client,RudaCount::Static(warps.div_ceil(4) as u32,1,1),
            RudaDim::new_1d(128),input.arg(),out.arg(),width,exponential)};
    }
    unsafe {scan::prepare::<CudaRuntime>(client,
        RudaCount::Static(u32::try_from(input.len.div_ceil(128)).expect("segment scan grid overflow"),1,1),
        RudaDim::new_1d(128),input.arg(),out.arg(),width,exponential)}
}
