use super::*;

#[ruda]
fn before<E:Numeric+RudaElement>(a:E,ai:u32,b:E,bi:u32,#[comptime] descending:bool)->bool {
    let an=a!=a;let bn=b!=b;
    let mut order=a<b;
    if comptime!(descending) {order=a>b;}
    if a==b || an && bn {order=ai<bi;}
    else if an!=bn {
        order=bn;
        if comptime!(descending) {order=an;}
    }
    order
}

#[ruda(launch)]
fn initialize<E:Numeric+RudaElement>(input:&Tensor<E>,values:&mut Tensor<E>,keys:&mut Tensor<u32>,
    length:usize,inner:usize,#[comptime] boolean:bool)
{
    let p=ABSOLUTE_POS as usize;
    if p<values.len() {
        let row=p/length;let index=p%length;
        let source=(row/inner)*length*inner+index*inner+row%inner;
        let mut value=input[source];
        if comptime!(boolean) {value=E::cast_from(value!=E::cast_from(0));}
        values[p]=value;keys[p]=u32::cast_from(index);
    }
}

#[ruda(launch)]
fn merge<E:Numeric+RudaElement>(values:&Tensor<E>,keys:&Tensor<u32>,out:&mut Tensor<E>,out_keys:&mut Tensor<u32>,
    length:usize,width:usize,#[comptime] descending:bool)
{
    let p=ABSOLUTE_POS as usize;
    if p<values.len() {
        let row=p/length;let position=p%length;let group=position/width;
        let start=group*width;let right=group%2!=0;
        let mut other_start=start+usize::min(width,length-start);
        let mut other_end=other_start+usize::min(width,length-other_start);
        let mut base=start;
        if right {other_start=start-width;other_end=start;base=other_start;}
        let value=values[p];let key=keys[p];let mut lo=other_start;let mut hi=other_end;
        while lo<hi {
            let mid=lo+(hi-lo)/2;let other=row*length+mid;
            if before::<E>(values[other],keys[other],value,key,descending) {lo=mid+1;}
            else {hi=mid;}
        }
        let destination=row*length+base+(position-start)+(lo-other_start);
        out[destination]=value;out_keys[destination]=key;
    }
}

#[ruda(launch)]
fn emit<E:Numeric+RudaElement>(values:&Tensor<E>,keys:&Tensor<u32>,out:&mut Tensor<E>,indices:&mut Tensor<i64>,
    length:usize,k:usize,inner:usize,#[comptime] rank:usize,#[comptime] indices_only:bool)
{
    let p=ABSOLUTE_POS as usize;
    if p<indices.len() {
        let row=(p/(k*inner))*inner+p%inner;let index=(p/inner)%k;
        let source=row*length+index;let mut coordinate=p;let mut vi=0usize;let mut ii=0usize;
        #[unroll]
        for axis in 0..rank {
            let d=comptime!(rank-axis-1);let pos=coordinate%indices.shape(d);coordinate/=indices.shape(d);
            ii+=pos*indices.stride(d);
            if comptime!(!indices_only) {vi+=pos*out.stride(d);}
        }
        indices[ii]=i64::cast_from(keys[source]);
        if comptime!(!indices_only) {out[vi]=values[source];}
    }
}

pub(super) fn validate(vs:&[View],ints:&[i64],scalars:&[f32]) {
    assert_eq!(ints.len(),7);assert!(scalars.is_empty());assert!((0..=2).contains(&ints[0]));
    assert_eq!(vs.len(),if ints[0]==2 {2}else {3});let input=&vs[0];
    assert!(input.dtype<=9 && input.shape.len()<=32 && input.shape.len()>0);
    if ints[0]==0 {assert_ne!(input.dtype,3,"topk does not support bool");}
    operation_graph::packed(input);
    assert!(ints[1]>=0 && (ints[1] as usize)<input.shape.len());
    for &flag in &ints[3..] {assert!(matches!(flag,0|1));}
    let axis=ints[1] as usize;let length=input.shape[axis];
    assert!(ints[2]>=0 && (ints[2] as usize)<=length);
    if ints[0]!=0 {assert_eq!(ints[2] as usize,length);}
    if ints[6]!=0 {assert_eq!(input.shape,vec![1]);assert_eq!(axis,0);}
    let mut shape=input.shape.clone();shape[axis]=if ints[6]!=0 {1}else {ints[2] as usize};
    for output in &vs[1..] {
        assert_eq!(output.shape,shape);assert!(output.len<=u32::MAX as usize);
        if output.len>0 {
            let mut axes:Vec<_>=output.shape.iter().zip(&output.strides).filter(|(size,_)|**size>1).collect();
            axes.sort_by_key(|(_,stride)|**stride);let mut extent=1usize;
            for (&size,&stride) in axes {assert_eq!(stride,extent,"sort output must be dense nonoverlapping storage");extent*=size;}
        }
    }
    if ints[0]!=2 {assert_eq!(vs[1].dtype,input.dtype);}
    assert_eq!(vs.last().unwrap().dtype,4);
}

fn allocate(c:&ComputeClient<CudaRuntime>,shape:&[usize],dtype:u32)->View {
    let size=shape.iter().try_fold(1usize,|n,&d|n.checked_mul(d)).expect("sort workspace overflow");
    View::packed(c.empty(size.max(1).checked_mul(element_bytes(dtype)).expect("sort workspace bytes overflow")),shape.to_vec(),dtype)
}

fn plan<E:Numeric+RudaElement>(c:&ComputeClient<CudaRuntime>,vs:&[View],ints:&[i64])->(Vec<PreparedKernel<CudaRuntime>>,Vec<View>) {
    let input=&vs[0];let axis=ints[1] as usize;let length=input.shape[axis];
    let inner=input.shape[axis+1..].iter().product::<usize>();let rows=input.len/length;
    let shape=[rows,length];let mut scratch=vec![allocate(c,&shape,input.dtype),allocate(c,&shape,5)];
    let grid=RudaCount::Static(u32::try_from(input.len.div_ceil(128)).unwrap(),1,1);let dim=RudaDim::new_1d(128);
    let mut nodes=vec![initialize::prepare::<E,CudaRuntime>(c,grid.clone(),dim,input.arg(),scratch[0].arg(),scratch[1].arg(),length,inner,input.dtype==3)];
    if length>1 {scratch.push(allocate(c,&shape,input.dtype));scratch.push(allocate(c,&shape,5));}
    let mut current=0;let mut width=1;
    while width<length {
        let next=2-current;
        nodes.push(merge::prepare::<E,CudaRuntime>(c,grid.clone(),dim,scratch[current].arg(),scratch[current+1].arg(),
            scratch[next].arg(),scratch[next+1].arg(),length,width,ints[3]!=0));
        current=next;width=width.checked_mul(2).expect("sort stage overflow");
    }
    let indices=vs.last().unwrap();let values=if ints[0]==2 {&scratch[current]}else {&vs[1]};
    let k=indices.shape[axis];let output_grid=RudaCount::Static(u32::try_from(indices.len.div_ceil(128)).unwrap(),1,1);
    nodes.push(emit::prepare::<E,CudaRuntime>(c,output_grid,dim,scratch[current].arg(),scratch[current+1].arg(),
        values.arg(),indices.arg(),length,k,inner,indices.shape.len(),ints[0]==2));
    (nodes,scratch)
}

pub(super) fn prepare(c:&ComputeClient<CudaRuntime>,vs:&[View],ints:&[i64])->(Vec<PreparedKernel<CudaRuntime>>,Vec<View>) {
    if vs.last().unwrap().len==0 {return (Vec::new(),Vec::new());}
    macro_rules! run {($e:ty)=>{plan::<$e>(c,vs,ints)};}
    match vs[0].dtype {0=>run!(f32),1=>run!(f16),2=>run!(bf16),3|8=>run!(u8),4=>run!(i64),5=>run!(i32),6=>run!(i16),7=>run!(i8),9=>run!(f64),_=>unreachable!()}
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_sorting_api_version()->u32 {1}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_sorting(descriptors:*const Descriptor,count:usize,integers:*const i64,integer_count:usize)->i32 {
    checked(|| {
        assert!((2..=3).contains(&count) && integer_count==7);
        let vs:Vec<_>=unsafe {abi::slice(descriptors,count)}.iter().map(|d|unsafe {abi::view(d)}).collect();
        let ints=unsafe {abi::slice(integers,integer_count)};validate(&vs,ints,&[]);
        let c=client().fixed_execution_queue();let (nodes,_workspace)=prepare(&c,&vs,ints);let launches=nodes.len();
        for node in nodes {let (task,count,arguments,queue)=node.into_parts();queue.launch(task,count,arguments);}
        LAUNCHES.fetch_add(launches as u64,Ordering::Relaxed);finish_dispatch(&c);
    })
}
