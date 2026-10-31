use super::*;

#[derive(Clone,Debug,Hash,PartialEq,Eq)]
struct Geometry {
    mode:u32,batch:usize,groups:usize,ic:usize,oc:usize,volume:usize,
    input:Vec<usize>,output:Vec<usize>,filter:Vec<usize>,
    stride:Vec<usize>,padding:Vec<usize>,dilation:Vec<usize>,
    a_shape:Vec<usize>,a_strides:Vec<usize>,b_shape:Vec<usize>,b_strides:Vec<usize>,
    target_shape:Vec<usize>,target_strides:Vec<usize>,
    a_packed:bool,b_packed:bool,target_packed:bool,
    input_size:usize,output_size:usize,m:usize,n:usize,k:usize,
}

fn indexable(value:&View)->bool {
    value.len<=u32::MAX as usize && value.shape.iter().zip(&value.strides).try_fold(0usize,|n,(&size,&stride)|
        n.checked_add(size.saturating_sub(1).checked_mul(stride)?)).is_some_and(|n|n<=u32::MAX as usize)
}

fn geometry(op:u32,a:&View,b:&View,out:&View,p:&[i64])->Geometry {
    let rank=a.shape.len();assert!((3..=5).contains(&rank));let dim=rank-2;
    assert_eq!(b.shape.len(),rank);assert_eq!(out.shape.len(),rank);assert_eq!(p.len(),4*dim+1);
    let groups=usize::try_from(p[4*dim]).expect("negative convolution groups");assert!(groups>0);
    let (input,gradient,weight)=match op {0=>(a,out,b),1|3=>(out,a,b),2=>(a,b,out),_=>unreachable!()};
    assert_eq!(input.shape[0],gradient.shape[0]);
    assert!(input.shape[1]>0 && gradient.shape[1]>0 && input.shape[1]%groups==0 && gradient.shape[1]%groups==0);
    let ic=input.shape[1]/groups;let oc=gradient.shape[1]/groups;
    assert_eq!(weight.shape[0],gradient.shape[1]);assert_eq!(weight.shape[1],ic);
    let parameter=|start|p[start..start+dim].iter().map(|&v|usize::try_from(v).expect("negative convolution parameter")).collect::<Vec<_>>();
    let stride=parameter(0);let padding=parameter(dim);let dilation=parameter(2*dim);
    for axis in 0..dim {
        assert!(stride[axis]>0 && dilation[axis]>0 && weight.shape[axis+2]>0);
        let extent=dilation[axis].checked_mul(weight.shape[axis+2]-1).and_then(|v|v.checked_add(1)).expect("convolution filter overflow");
        if op==3 || op==2 && p[3*dim+axis]!=0 {
            let extra=usize::try_from(p[3*dim+axis]).expect("negative output padding");
            assert!(extra<stride[axis] || extra<dilation[axis]);
            let size=gradient.shape[axis+2].checked_sub(1).and_then(|v|v.checked_mul(stride[axis]))
                .and_then(|v|v.checked_add(extent)).and_then(|v|v.checked_add(extra))
                .and_then(|v|v.checked_sub(padding[axis].checked_mul(2)?)).expect("transposed convolution extent overflow");
            assert_eq!(input.shape[axis+2],size);
        } else {
            let padded=input.shape[axis+2].checked_add(padding[axis].checked_mul(2).expect("padding overflow")).expect("convolution extent overflow");
            assert!(padded>=extent);assert_eq!(gradient.shape[axis+2],(padded-extent)/stride[axis]+1);
        }
    }
    let input_size=input.shape[2..].iter().product::<usize>();let output_size=gradient.shape[2..].iter().product::<usize>();
    let volume=weight.shape[2..].iter().product::<usize>();let batch=input.shape[0];
    let (m,n,k)=match op {0=>(oc,batch*output_size,ic*volume),1|3=>(ic,batch*input_size,oc*volume),2=>(oc,ic*volume,batch*output_size),_=>unreachable!()};
    Geometry{mode:if op==3 {1}else {op},batch,groups,ic,oc,volume,input:input.shape[2..].to_vec(),output:gradient.shape[2..].to_vec(),
        filter:weight.shape[2..].to_vec(),stride,padding,dilation,a_shape:a.shape.clone(),a_strides:a.strides.clone(),
        b_shape:b.shape.clone(),b_strides:b.strides.clone(),target_shape:out.shape.clone(),target_strides:out.strides.clone(),
        a_packed:a.contiguous(),b_packed:b.contiguous(),target_packed:out.contiguous(),input_size,output_size,m,n,k}
}

#[ruda]
fn offset(position:usize,#[comptime] shape:Vec<usize>,#[comptime] strides:Vec<usize>,#[comptime] packed:bool)->usize {
    let mut result=position;
    if comptime!(!packed) {
        result=0;let mut remaining=position;
        #[unroll] for reverse in 0usize..shape.len() {
            let d=comptime!(shape.len()-1-reverse);
            result+=(remaining%comptime!(shape[d]))*comptime!(strides[d]);remaining/=comptime!(shape[d]);
        }
    }
    result
}

#[ruda]
fn patch(position:usize,tap:usize,#[comptime] cfg:Geometry,#[comptime] inverse:bool)->(usize,bool) {
    let mut pos=position;let mut filter=tap;let mut address=0usize;let mut multiplier=1usize;let mut valid=true;
    #[unroll] for reverse in 0usize..cfg.input.len() {
        let d=comptime!(cfg.input.len()-1-reverse);let kernel=filter%comptime!(cfg.filter[d]);filter/=comptime!(cfg.filter[d]);
        let mut coordinate=0i64;
        if comptime!(inverse) {
            let here=pos%comptime!(cfg.input[d]);pos/=comptime!(cfg.input[d]);
            let numerator=i64::cast_from(here)+comptime!(cfg.padding[d]) as i64-i64::cast_from(kernel)*comptime!(cfg.dilation[d]) as i64;
            valid=valid && numerator>=0i64 && numerator%comptime!(cfg.stride[d]) as i64==0i64;
            coordinate=numerator/comptime!(cfg.stride[d]) as i64;
            valid=valid && coordinate<i64::cast_from(comptime!(cfg.output[d]));
            if coordinate>=0i64 {address+=usize::cast_from(coordinate)*multiplier;}multiplier*=comptime!(cfg.output[d]);
        } else {
            let here=pos%comptime!(cfg.output[d]);pos/=comptime!(cfg.output[d]);
            coordinate=i64::cast_from(here)*comptime!(cfg.stride[d]) as i64-comptime!(cfg.padding[d]) as i64
                +i64::cast_from(kernel)*comptime!(cfg.dilation[d]) as i64;
            valid=valid && coordinate>=0i64 && coordinate<i64::cast_from(comptime!(cfg.input[d]));
            if coordinate>=0i64 {address+=usize::cast_from(coordinate)*multiplier;}multiplier*=comptime!(cfg.input[d]);
        }
    }
    (address,valid)
}

#[ruda]
fn left<F:Float>(b:&Tensor<F>,group:usize,row:usize,reduction:usize,#[comptime] cfg:Geometry)->F {
    let mut position=0usize;
    if comptime!(cfg.mode==0) {position=((group*cfg.oc+row)*cfg.ic+reduction%cfg.ic)*cfg.volume+reduction/cfg.ic;}
    else if comptime!(cfg.mode==1) {position=((group*cfg.oc+reduction/cfg.volume)*cfg.ic+row)*cfg.volume+reduction%cfg.volume;}
    else {position=((reduction%cfg.batch)*(cfg.groups*cfg.oc)+group*cfg.oc+row)*cfg.output_size+reduction/cfg.batch;}
    b[offset(position,cfg.b_shape,cfg.b_strides,cfg.b_packed)]
}

#[ruda]
fn right<F:Float>(a:&Tensor<F>,group:usize,column:usize,reduction:usize,#[comptime] cfg:Geometry)->F {
    let mut position=0usize;let mut valid=false;
    if comptime!(cfg.mode==0) {
        let mapped=patch(column%cfg.output_size,reduction/cfg.ic,cfg,false);valid=mapped.1;
        position=((column/cfg.output_size)*(cfg.groups*cfg.ic)+group*cfg.ic+reduction%cfg.ic)*cfg.input_size+mapped.0;
    } else if comptime!(cfg.mode==1) {
        let mapped=patch(column%cfg.input_size,reduction%cfg.volume,cfg,true);valid=mapped.1;
        position=((column/cfg.input_size)*(cfg.groups*cfg.oc)+group*cfg.oc+reduction/cfg.volume)*cfg.output_size+mapped.0;
    } else {
        let mapped=patch(reduction/cfg.batch,column%cfg.volume,cfg,false);valid=mapped.1;
        position=((reduction%cfg.batch)*(cfg.groups*cfg.ic)+group*cfg.ic+column/cfg.volume)*cfg.input_size+mapped.0;
    }
    let mut value=F::cast_from(0);if valid {value=a[offset(position,cfg.a_shape,cfg.a_strides,cfg.a_packed)];}value
}

#[ruda]
fn destination(group:usize,row:usize,column:usize,#[comptime] cfg:Geometry)->usize {
    let mut position=0usize;
    if comptime!(cfg.mode==2) {position=(group*cfg.oc+row)*cfg.n+column;}
    else {
        let spatial=comptime!(if cfg.mode==0 {cfg.output_size}else {cfg.input_size});
        position=((column/spatial)*(cfg.groups*cfg.m)+group*cfg.m+row)*spatial+column%spatial;
    }
    position
}

#[ruda]
fn load_tile<F:Float>(a:&Tensor<F>,b:&Tensor<F>,lhs:&mut SharedMemory<F>,rhs:&mut SharedMemory<F>,
    group:usize,row:usize,column:usize,base:usize,end:usize,#[comptime] cfg:Geometry,#[comptime] tile_m:usize)
{
    let tid=UNIT_POS as usize;
    #[unroll] for step in 0usize..comptime!(tile_m/4) {
        let p=tid+step*128;let r=row+p/32;let k=base+p%32;let mut value=F::cast_from(0);
        if r<cfg.m && k<end {value=left(b,group,r,k,cfg);}lhs[p]=value;
    }
    if comptime!((cfg.mode==0 && cfg.ic%32==0) || (cfg.mode==1 && cfg.volume==1)
        || (cfg.mode==2 && cfg.batch%32==0)) {
        let col=column+tid%32;let mut source=0usize;let mut valid=false;
        let channel_stride=comptime!(if cfg.mode==1 {cfg.output_size}
            else if cfg.mode==2 {cfg.groups*cfg.ic*cfg.input_size}else {cfg.input_size});
        if col<cfg.n && base<end {
            if comptime!(cfg.mode==0) {
                let mapped=patch(col%cfg.output_size,base/cfg.ic,cfg,false);valid=mapped.1;
                source=((col/cfg.output_size)*(cfg.groups*cfg.ic)+group*cfg.ic+base%cfg.ic)*cfg.input_size+mapped.0;
            } else if comptime!(cfg.mode==1) {
                let mapped=patch(col%cfg.input_size,0usize,cfg,true);valid=mapped.1;
                source=((col/cfg.input_size)*(cfg.groups*cfg.oc)+group*cfg.oc+base)*cfg.output_size+mapped.0;
            } else {
                let mapped=patch(base/cfg.batch,col%cfg.volume,cfg,false);valid=mapped.1;
                source=((base%cfg.batch)*(cfg.groups*cfg.ic)+group*cfg.ic+col/cfg.volume)*cfg.input_size+mapped.0;
            }
        }
        #[unroll] for step in 0usize..8 {
            let p=tid+step*128;let k=base+p/32;let mut value=F::cast_from(0);
            if valid && k<end {
                value=a[offset(source+(p/32)*channel_stride,cfg.a_shape,cfg.a_strides,cfg.a_packed)];
            }
            rhs[p]=value;
        }
    } else {
        #[unroll] for step in 0usize..8 {
            let p=tid+step*128;let k=base+p/32;let col=column+p%32;let mut value=F::cast_from(0);
            if col<cfg.n && k<end {value=right(a,group,col,k,cfg);}rhs[p]=value;
        }
    }
}

#[ruda(launch)]
fn implicit<F:Float+RudaElement>(a:&Tensor<F>,b:&Tensor<F>,out:&mut Tensor<f32>,
    #[comptime] cfg:Geometry,#[comptime] tile_m:usize,#[comptime] accelerated:bool,
    #[comptime] partitions:usize,#[comptime] partition_k:usize)
{
    let tid=UNIT_POS as usize;let warp=tid/32;let part=RUDA_POS_Z as usize%partitions;let group=RUDA_POS_Z as usize/partitions;
    let row=RUDA_POS_Y as usize*tile_m;let column=RUDA_POS_X as usize*32;
    let mut lhs=SharedMemory::<F>::new_aligned(tile_m*32,32usize);let mut rhs=SharedMemory::<F>::new_aligned(1024usize,32usize);
    let mut result=SharedMemory::<f32>::new_aligned(tile_m*32,32usize);
    let mut base=part*partition_k;let end=usize::min(base+partition_k,cfg.k);
    if comptime!(accelerated) {
        let acc=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,16usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0f32);
        while base<end {
            load_tile(a,b,&mut lhs,&mut rhs,group,row,column,base,end,cfg,tile_m);sync_ruda();
            if warp<comptime!(tile_m/8) {
                #[unroll] for half in 0usize..2 {
                    let aa=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A,16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,
                        &lhs.slice((warp/2)*512+half*16,tile_m*32),32u32);
                    let bb=cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B,16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,
                        &rhs.slice(half*512+(warp%2)*16,1024usize),32u32);
                    cmma::execute::<F,F,f32,f32,_>(&aa,&bb,&acc,&acc);
                }
            }
            sync_ruda();if end-base<=32 {base=end;}else {base+=32;}
        }
        if warp<comptime!(tile_m/8) {cmma::store(&mut result.slice_mut((warp/2)*512+(warp%2)*16,tile_m*32),&acc,32u32,cmma::MatrixLayout::RowMajor);}
    } else {
        let mut sums=Array::<f32>::new(tile_m/4);#[unroll] for i in 0usize..comptime!(tile_m/4) {sums[i]=0.0f32;}
        while base<end {
            load_tile(a,b,&mut lhs,&mut rhs,group,row,column,base,end,cfg,tile_m);sync_ruda();
            #[unroll] for k in 0usize..32 {
                #[unroll] for item in 0usize..comptime!(tile_m/4) {
                    let p=tid+item*128;let x=f32::cast_from(lhs[(p/32)*32+k]);let y=f32::cast_from(rhs[k*32+p%32]);
                    sums[item]+=x*y;
                }
            }
            sync_ruda();if end-base<=32 {base=end;}else {base+=32;}
        }
        #[unroll] for i in 0usize..comptime!(tile_m/4) {result[tid+i*128]=sums[i];}
    }
    sync_ruda();
    #[unroll] for item in 0usize..comptime!(tile_m/4) {
        let p=tid+item*128;let r=row+p/32;let col=column+p%32;
        if r<cfg.m && col<cfg.n {
            let logical=destination(group,r,col,cfg);let mut at=0usize;
            if comptime!(partitions>1) {at=part*(cfg.groups*cfg.m*cfg.n)+logical;}
            else {at=offset(logical,cfg.target_shape,cfg.target_strides,cfg.target_packed);}
            out[at]=result[p];
        }
    }
}

#[ruda(launch)]
fn direct<F:Float>(a:&Tensor<F>,b:&Tensor<F>,out:&mut Tensor<f32>,#[comptime] cfg:Geometry) {
    let p=ABSOLUTE_POS as usize;
    if p<cfg.groups*cfg.m*cfg.n {
        let group=p/(cfg.m*cfg.n);let row=p/cfg.n%cfg.m;let column=p%cfg.n;let mut value=0.0f32;
        for k in 0usize..cfg.k {value+=f32::cast_from(left(b,group,row,k,cfg))*f32::cast_from(right(a,group,column,k,cfg));}
        let logical=destination(group,row,column,cfg);out[offset(logical,cfg.target_shape,cfg.target_strides,cfg.target_packed)]=value;
    }
}

#[ruda(launch)]
fn zero(out:&mut Tensor<f32>,#[comptime] shape:Vec<usize>,#[comptime] strides:Vec<usize>,#[comptime] packed:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {out[offset(p,shape,strides,packed)]=0.0f32;}
}

fn launch_direct(c:&ComputeClient<CudaRuntime>,a:&View,b:&View,out:&View,cfg:Geometry)->bool {
    let h=&c.properties().hardware;
    let threads=128u32.min(h.max_units_per_ruda).min(h.max_ruda_dim.0).max(1);
    let grid=RudaCount::Static(u32::try_from(out.len.div_ceil(threads as usize)).expect("convolution grid overflow"),1,1);
    if cfg.k==0 {
        unsafe {zero::launch::<CudaRuntime>(c,grid,RudaDim::new_1d(threads),out.arg(),cfg.target_shape,cfg.target_strides,cfg.target_packed);}
    } else {
        macro_rules! run {($f:ty)=>{unsafe {direct::launch::<$f,CudaRuntime>(c,grid,RudaDim::new_1d(threads),a.arg(),b.arg(),out.arg(),cfg)}};}
        match a.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),_=>unreachable!()}
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(c);true
}

#[ruda(launch)]
fn depthwise_weight<F:Float>(a:&Tensor<F>,b:&Tensor<F>,out:&mut Tensor<f32>,#[comptime] cfg:Geometry) {
    let p=RUDA_POS_X as usize;let tid=UNIT_POS as usize;
    let group=p/cfg.n;let column=p%cfg.n;let mut k=tid;let mut sum=0.0f32;
    while k<cfg.k {
        sum+=f32::cast_from(left(b,group,0usize,k,cfg))*f32::cast_from(right(a,group,column,k,cfg));
        if cfg.k-k<=256usize {k=cfg.k;}else {k+=256usize;}
    }
    sum=plane_sum(sum);let mut partials=SharedMemory::<f32>::new(8usize);
    if tid%32==0 {partials[tid/32]=sum;}sync_ruda();
    if tid<32 {
        let mut value=0.0f32;if tid<8 {value=partials[tid];}value=plane_sum(value);
        if tid==0 {out[offset(p,cfg.target_shape,cfg.target_strides,cfg.target_packed)]=value;}
    }
}

#[ruda(launch)]
fn merge(input:&Tensor<f32>,out:&mut Tensor<f32>,#[comptime] cfg:Geometry,#[comptime] partitions:usize) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let mut value=0.0f32;#[unroll] for part in 0usize..partitions {value+=input[part*out.len()+p];}
        out[offset(p,cfg.target_shape,cfg.target_strides,cfg.target_packed)]=value;
    }
}

pub(super) fn launch(op:u32,a:&View,b:&View,out:&View,p:&[i64])->bool {
    if op>3 || a.dtype>2 || a.dtype!=b.dtype || out.dtype!=0 || ![a,b,out].iter().all(|v|indexable(v)) {return false;}
    let cfg=geometry(op,a,b,out,p);
    if out.len==0 {return true;}
    let c=client();let h=&c.properties().hardware;
    if cfg.k==0 {return launch_direct(&c,a,b,out,cfg);}
    if cfg.mode==2 && cfg.ic==1 && cfg.oc==1 && cfg.k>=256 && h.max_units_per_ruda>=256 && h.max_ruda_dim.0>=256
        && h.plane_size_min==32 && h.plane_size_max==32 && out.len<=h.max_ruda_count.0 as usize {
        macro_rules! run {($f:ty)=>{unsafe {depthwise_weight::launch::<$f,CudaRuntime>(&c,RudaCount::Static(out.len as u32,1,1),
            RudaDim::new_1d(256),a.arg(),b.arg(),out.arg(),cfg)}};}
        match a.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),_=>unreachable!()}
        LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);return true;
    }
    if a.dtype==0 && cfg.ic==1 && cfg.oc==1 {return false;}
    let small=cfg.m<8 && cfg.n<16 || cfg.ic==1 && cfg.oc==1 || h.max_units_per_ruda<128 || h.max_ruda_dim.0<128;
    if small {return launch_direct(&c,a,b,out,cfg);}
    use ruda_core::ir::{ElemType,FloatKind,features::MmaConfig};
    let storage=ElemType::Float(if a.dtype==1 {FloatKind::F16}else {FloatKind::BF16}).into();
    let mma=MmaConfig{a_type:storage,b_type:storage,cd_type:ElemType::Float(FloatKind::F32).into(),m:16,n:16,k:16};
    let accelerated=a.dtype!=0 && c.features().matmul.cmma.contains(&mma) && h.plane_size_min==32 && h.plane_size_max==32;
    let tile_m=if cfg.m<=8 && !accelerated {8}else if cfg.m<=16 {16}else {32};
    let shared=(tile_m*32+1024)*element_bytes(a.dtype)+tile_m*32*4;
    if shared>h.max_shared_memory_size as usize {return launch_direct(&c,a,b,out,cfg);}
    let tiles=cfg.m.div_ceil(tile_m)*cfg.n.div_ceil(32)*cfg.groups;
    let sms=h.num_streaming_multiprocessors.unwrap_or(108).max(1) as usize;
    let mut partitions=if cfg.mode==2 && cfg.k>=2048 {(2*sms).div_ceil(tiles).min(cfg.k.div_ceil(1024)).min(16).max(1)}else {1};
    while partitions>1 && (out.len.checked_mul(partitions).and_then(|v|v.checked_mul(4)).is_none_or(|v|v>64*1024*1024)
        || cfg.groups*partitions>h.max_ruda_count.2 as usize) {partitions-=1;}
    let partition_k=cfg.k.div_ceil(partitions).div_ceil(32)*32;
    if partition_k>u32::MAX as usize || cfg.n.div_ceil(32)>h.max_ruda_count.0 as usize || cfg.m.div_ceil(tile_m)>h.max_ruda_count.1 as usize
        || cfg.groups*partitions>h.max_ruda_count.2 as usize {return launch_direct(&c,a,b,out,cfg);}
    let scratch=if partitions>1 {Some(View::packed(c.empty(out.len*partitions*4),vec![partitions,out.len],0))}else {None};
    let target=scratch.as_ref().unwrap_or(out);let grid=RudaCount::Static(cfg.n.div_ceil(32) as u32,cfg.m.div_ceil(tile_m) as u32,(cfg.groups*partitions) as u32);
    macro_rules! run {($f:ty)=>{unsafe {implicit::launch::<$f,CudaRuntime>(&c,grid,RudaDim::new_1d(128),a.arg(),b.arg(),target.arg(),cfg.clone(),tile_m,accelerated,partitions,partition_k)}};}
    match a.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),_=>unreachable!()}
    if let Some(scratch)=scratch {unsafe {merge::launch::<CudaRuntime>(&c,RudaCount::Static(u32::try_from(out.len.div_ceil(128)).unwrap(),1,1),
        RudaDim::new_1d(128),scratch.arg(),out.arg(),cfg,partitions);}}
    LAUNCHES.fetch_add(if partitions>1 {2}else {1},Ordering::Relaxed);finish_dispatch(&c);true
}

#[ruda(launch)]
fn bias<F:Float>(input:&Tensor<F>,out:&mut Tensor<f32>,#[comptime] shape:Vec<usize>,#[comptime] strides:Vec<usize>,#[comptime] width:usize) {
    let channel=RUDA_POS_X as usize;let lane=UNIT_POS as usize;let mut sum=0.0f32;let mut i=lane;
    while i<width {
        let spatial=comptime!(shape[2..].iter().product::<usize>());let batch=i/spatial;let mut position=i%spatial;
        let mut at=batch*comptime!(strides[0])+channel*comptime!(strides[1]);
        #[unroll] for reverse in 0usize..comptime!(shape.len()-2) {
            let d=comptime!(shape.len()-1-reverse);at+=(position%comptime!(shape[d]))*comptime!(strides[d]);position/=comptime!(shape[d]);
        }
        sum+=f32::cast_from(input[at]);if width-i<=256usize {i=width;}else {i+=256usize;}
    }
    sum=plane_sum(sum);let mut partials=SharedMemory::<f32>::new(8usize);
    if lane%32==0 {partials[lane/32]=sum;}sync_ruda();
    if lane<32 {let mut value=0.0f32;if lane<8 {value=partials[lane];}value=plane_sum(value);if lane==0 {out[channel*out.stride(0)]=value;}}
}

pub(super) fn bias_backward(a:&View,out:&View)->bool {
    if a.dtype>2 || out.dtype!=0 || !indexable(a) || !indexable(out) {return false;}
    assert!((3..=5).contains(&a.shape.len()));assert_eq!(out.shape,vec![a.shape[1]]);
    if out.len==0 {return true;}
    let c=client();let grid=RudaCount::Static(u32::try_from(out.len).unwrap(),1,1);let width=a.len/out.len;
    if width==0 {
        unsafe {zero::launch::<CudaRuntime>(&c,RudaCount::Static(u32::try_from(out.len.div_ceil(128)).unwrap(),1,1),
            RudaDim::new_1d(128),out.arg(),out.shape.clone(),out.strides.clone(),out.contiguous());}
    } else {
        macro_rules! run {($f:ty)=>{unsafe {bias::launch::<$f,CudaRuntime>(&c,grid,RudaDim::new_1d(256),a.arg(),out.arg(),a.shape.clone(),a.strides.clone(),width)}};}
        match a.dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),_=>unreachable!()}
    }
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);true
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_spatial_accumulate_api_version()->u32 {1}
