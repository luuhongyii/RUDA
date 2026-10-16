use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn bag_range<I:Numeric+RudaElement>(offsets:&Tensor<I>,bag:usize,bags:usize,samples:usize)->(usize,usize) {
    let mut begin=0i64;let mut end=i64::cast_from(samples);
    if bag!=0 {begin=i64::cast_from(offsets[address(offsets,bag)]);}
    if bag+1<bags {end=i64::cast_from(offsets[address(offsets,bag+1)]);}
    if begin<0 || end<begin || end>i64::cast_from(samples) {
        intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
    }
    (usize::cast_from(begin),usize::cast_from(end))
}

#[ruda]
fn row_index<I:Numeric+RudaElement>(indices:&Tensor<I>,sample:usize,rows:usize)->i64 {
    let row=i64::cast_from(indices[address(indices,sample)]);
    if row<0 || row>=i64::cast_from(rows) {
        intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
    }
    row
}

#[ruda(launch)]
fn metadata<I:Numeric+RudaElement>(indices:&Tensor<I>,offsets:&Tensor<I>,mapping:&mut Tensor<I>,sizes:&mut Tensor<I>,bags:usize,rows:usize,padding:i64) {
    let bag=ABSOLUTE_POS as usize;
    if bag<bags {
        let (begin,end)=bag_range(offsets,bag,bags,indices.len());let mut count=0usize;
        for sample in begin..end {
            let row=row_index(indices,sample,rows);
            if row!=padding {count+=1;}
            mapping[address(mapping,sample)]=I::cast_from(bag);
        }
        sizes[address(sizes,bag)]=I::cast_from(count);
    }
}

#[ruda(launch)]
fn forward<S:Float+RudaElement,F:Float+RudaElement,I:Numeric+RudaElement>(weight:&Tensor<S>,indices:&Tensor<I>,offsets:&Tensor<I>,sample_weights:&Tensor<S>,out:&mut Tensor<S>,maximum:&mut Tensor<I>,padding:i64,#[comptime] mode:u32,#[comptime] weighted:bool) {
    let index=ABSOLUTE_POS as usize;
    if index<out.len() {
        let width=out.shape(1);let bag=index/width;let feature=index%width;
        let (begin,end)=bag_range(offsets,bag,out.shape(0),indices.len());
        let mut count=0usize;let mut value=F::new(0.0);let mut max_row=-1i64;
        for sample in begin..end {
            let row=row_index(indices,sample,weight.shape(0));
            let mut item=F::cast_from(weight[address(weight,usize::cast_from(row)*width+feature)]);
            if comptime!(mode==2) {
                if row!=padding {
                    if count==0 || item>value {value=item;max_row=row;}
                    count+=1;
                }
            } else {
                if row==padding {item=F::new(0.0);}else{count+=1;}
                if comptime!(weighted) {item*=F::cast_from(sample_weights[address(sample_weights,sample)]);}
                value+=item;
            }
        }
        if comptime!(mode==1) {if count!=0 {value/=F::cast_from(count);}}
        if comptime!(mode==2) {maximum[address(maximum,index)]=I::cast_from(max_row);}
        out[address(out,index)]=S::cast_from(value);
    }
}

#[ruda]
fn bound<I:Numeric+RudaElement>(indices:&Tensor<I>,row:i64,#[comptime] upper:bool)->usize {
    let mut low=0usize;let mut high=indices.len();
    while low<high {
        let middle=low+(high-low)/2;let key=i64::cast_from(indices[address(indices,middle)]);
        let mut advance=key<row;
        if comptime!(upper) {advance=key<=row;}
        if advance {low=middle+1;}else{high=middle;}
    }
    low
}

#[ruda(launch)]
fn dense_backward<S:Float+RudaElement,F:Float+RudaElement,I:Numeric+RudaElement>(grad:&Tensor<S>,sorted:&Tensor<I>,order:&Tensor<i64>,mapping:&Tensor<I>,sizes:&Tensor<I>,sample_weights:&Tensor<S>,out:&mut Tensor<S>,padding:i64,#[comptime] mean:bool,#[comptime] frequency:bool,#[comptime] weighted:bool) {
    let index=ABSOLUTE_POS as usize;
    if index<out.len() {
        let width=out.shape(1);let row=i64::cast_from(index/width);let feature=index%width;
        let mut total=F::new(0.0);
        if row!=padding {
            let begin=bound(sorted,row,false);let end=bound(sorted,row,true);
            let mut chunk=begin;
            while chunk<end {
                let mut stop=end;if end-chunk>10 {stop=chunk+10;}
                let mut partial=F::new(0.0);
                for position in chunk..stop {
                    let sample=usize::cast_from(order[address(order,position)]);
                    let bag=i64::cast_from(mapping[address(mapping,sample)]);
                    if bag<0 || bag>=i64::cast_from(grad.shape(0)) {
                        intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
                    }
                    let mut scale=F::new(1.0);
                    if comptime!(frequency) {scale=F::cast_from(1.0f64/f64::cast_from(end-begin));}
                    if comptime!(weighted) {scale*=F::cast_from(sample_weights[address(sample_weights,sample)]);}
                    let mut value=F::cast_from(grad[address(grad,usize::cast_from(bag)*width+feature)]);
                    if comptime!(mean) {value/=F::cast_from(sizes[address(sizes,usize::cast_from(bag))]);}
                    partial+=value*scale;
                }
                total+=partial;chunk=stop;
            }
        }
        out[address(out,index)]=S::cast_from(total);
    }
}

#[ruda(launch)]
fn max_backward<S:Float+RudaElement,I:Numeric+RudaElement>(grad:&Tensor<S>,maximum:&Tensor<I>,out:&mut Tensor<Atomic<S>>,padding:i64) {
    let index=ABSOLUTE_POS as usize;
    if index<grad.len() {
        let row=i64::cast_from(maximum[address(maximum,index)]);
        if row>=0 && row!=padding {
            if row>=i64::cast_from(out.shape(0)) {
                intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
            }
            let width=grad.shape(1);let feature=index%width;
            let target=usize::cast_from(row)*out.stride(0)+feature*out.stride(1);
            out[target].fetch_add(grad[address(grad,index)]);
        }
    }
}

#[ruda(launch)]
fn sample_backward<S:Float+RudaElement,F:Float+RudaElement,I:Numeric+RudaElement>(grad:&Tensor<S>,weight:&Tensor<S>,indices:&Tensor<I>,mapping:&Tensor<I>,out:&mut Tensor<S>,padding:i64) {
    let lane=UNIT_POS as usize%32;let sample=RUDA_POS_X as usize*8+UNIT_POS as usize/32;
    if sample<indices.len() {
        let row=row_index(indices,sample,weight.shape(0));let mut value=F::new(0.0);
        if row!=padding {
            let bag=i64::cast_from(mapping[address(mapping,sample)]);
            if bag<0 || bag>=i64::cast_from(grad.shape(0)) {
                intrinsic!(|scope| {scope.register(ruda_core::ir::Branch::Unreachable);});
            }
            let width=weight.shape(1);let mut feature=lane;
            while feature<width {
                let left=grad[address(grad,usize::cast_from(bag)*width+feature)];
                let right=weight[address(weight,usize::cast_from(row)*width+feature)];
                value+=F::cast_from(left*right);feature+=32;
            }
        }
        let mut offset=16u32;
        while offset>0 {value+=plane_shuffle_down(value,offset);offset/=2;}
        if lane==0 {out[address(out,sample)]=S::cast_from(value);}
    }
}

#[ruda(launch)]
fn restore_mapping<I:Numeric+RudaElement>(offsets:&Tensor<I>,out:&mut Tensor<I>) {
    let sample=ABSOLUTE_POS as usize;
    if sample<out.len() {
        let position=bound(offsets,i64::cast_from(sample),true);
        out[address(out,sample)]=I::cast_from(i64::cast_from(position)-1);
    }
}

fn grid(n:usize)->RudaCount {RudaCount::Static(u32::try_from(n.div_ceil(256)).expect("embedding bag launch overflow"),1,1)}

pub(super) fn dispatch(op:u32,xs:&[View],p:&[i64]) {
    let c=client().fixed_execution_queue();let block=RudaDim::new_1d(256);
    if op==344 {
        assert_eq!(xs.len(),2);assert!(p.is_empty());assert!(matches!(xs[0].dtype,4|5));
        assert_eq!(xs[0].dtype,xs[1].dtype);assert_eq!(xs[0].shape.len(),1);assert_eq!(xs[1].shape.len(),1);
        if xs[1].len==0 {return;}
        macro_rules! map {($i:ty)=>{unsafe {restore_mapping::launch::<$i,CudaRuntime>(&c,grid(xs[1].len),block,xs[0].arg(),xs[1].arg());}};}
        if xs[0].dtype==4 {map!(i64);}else{map!(i32);}
        LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);return;
    }
    assert!((341..=343).contains(&op));assert_eq!(p.len(),4);
    let mode=u32::try_from(p[0]).unwrap();assert!(mode<=2);let padding=p[1];
    assert!(matches!(p[2],0|1));assert!(matches!(p[3],0|1));let frequency=p[2]!=0;let weighted=p[3]!=0;
    assert!(!weighted || mode==0);assert!(matches!(xs[0].dtype,0|1|2|9));
    let scalar=xs[0].dtype;let index_type;
    if op==341 {
        assert_eq!(xs.len(),8);assert_eq!(xs[0].shape.len(),2);assert_eq!(xs[4].shape.len(),2);
        index_type=xs[1].dtype;
        for &i in &[1usize,2,5,6,7] {assert_eq!(xs[i].dtype,index_type);}
        for &i in &[3usize,4] {assert_eq!(xs[i].dtype,scalar);}
        for &i in &[1usize,2,5,6] {assert_eq!(xs[i].shape.len(),1);}
        assert_eq!(xs[4].shape[1],xs[0].shape[1]);assert_eq!(xs[5].len,xs[1].len);
        let bags=xs[4].shape[0];assert!(bags<=xs[2].len);assert_eq!(xs[6].len,xs[2].len);
        if mode==2 {assert_eq!(xs[7].shape,xs[4].shape);}else{assert_eq!(xs[7].len,0);}
        if weighted {assert_eq!(xs[3].len,xs[1].len);}
    } else if op==342 {
        assert_eq!(xs.len(),8);assert_eq!(xs[0].shape.len(),2);assert_eq!(xs[7].shape.len(),2);
        index_type=xs[1].dtype;assert_eq!(xs[2].dtype,4);
        for &i in &[3usize,4,5] {assert_eq!(xs[i].dtype,index_type);}
        for &i in &[6usize,7] {assert_eq!(xs[i].dtype,scalar);}
        assert_eq!(xs[7].shape[1],xs[0].shape[1]);
        if mode==2 {assert_ne!(scalar,2);assert_eq!(xs[5].shape,xs[0].shape);}
        else {
            assert_eq!(xs[1].shape.len(),1);assert_eq!(xs[2].len,xs[1].len);assert_eq!(xs[3].len,xs[1].len);
            if mode==1 {assert!(xs[4].len>=xs[0].shape[0]);}if weighted {assert_eq!(xs[6].len,xs[1].len);}
        }
    } else {
        assert_eq!(xs.len(),5);assert_eq!(mode,0);index_type=xs[2].dtype;
        assert_eq!(xs[3].dtype,index_type);assert_eq!(xs[1].dtype,scalar);assert_eq!(xs[4].dtype,scalar);
        assert_eq!(xs[0].shape.len(),2);assert_eq!(xs[1].shape.len(),2);assert_eq!(xs[0].shape[1],xs[1].shape[1]);
        assert_eq!(xs[2].shape.len(),1);assert_eq!(xs[3].len,xs[2].len);assert_eq!(xs[4].len,xs[2].len);
    }
    assert!(matches!(index_type,4|5));let mut launches=0u64;
    macro_rules! run {($s:ty,$f:ty,$i:ty)=>{unsafe {
        if op==341 {
            let bags=xs[4].shape[0];
            if bags!=0 {
                metadata::launch::<$i,CudaRuntime>(&c,grid(bags),block,xs[1].arg(),xs[2].arg(),xs[5].arg(),xs[6].arg(),bags,xs[0].shape[0],padding);launches+=1;
            }
            if xs[4].len!=0 {
                forward::launch::<$s,$f,$i,CudaRuntime>(&c,grid(xs[4].len),block,xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[7].arg(),padding,mode,weighted);launches+=1;
            }
        } else if op==342 {
            if mode==2 {
                if xs[0].len!=0 {max_backward::launch::<$s,$i,CudaRuntime>(&c,grid(xs[0].len),block,xs[0].arg(),xs[5].arg(),xs[7].arg(),padding);launches+=1;}
            } else if xs[7].len!=0 {
                dense_backward::launch::<$s,$f,$i,CudaRuntime>(&c,grid(xs[7].len),block,xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[6].arg(),xs[7].arg(),padding,mode==1,frequency,weighted);launches+=1;
            }
        } else if xs[4].len!=0 {
            let count=RudaCount::Static(u32::try_from(xs[4].len.div_ceil(8)).unwrap(),1,1);
            sample_backward::launch::<$s,$f,$i,CudaRuntime>(&c,count,block,xs[0].arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),padding);launches+=1;
        }
    }};}
    macro_rules! indexed {($s:ty,$f:ty)=>{if index_type==4 {run!($s,$f,i64);}else{run!($s,$f,i32);}};}
    match scalar {0=>indexed!(f32,f32),1=>indexed!(f16,f32),2=>indexed!(bf16,f32),9=>indexed!(f64,f64),_=>unreachable!()}
    LAUNCHES.fetch_add(launches,Ordering::Relaxed);finish_dispatch(&c);
}
