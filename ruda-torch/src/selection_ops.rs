use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::Tensor;

#[ruda]
fn radix_key<E:Numeric+RudaElement>(value:E,#[comptime] code:u32,#[comptime] bits:u32)->u64 {
    let mut key=0u64;
    if comptime!(code==0 || code==1 || code==2 || code==9) {
        if comptime!(code==9) {key=u64::reinterpret(value);}
        else if comptime!(code==0) {key=u64::cast_from(u32::reinterpret(value));}
        else {key=u64::cast_from(u16::reinterpret(value));}
        let sign=1u64<<(bits-1);let full=sign|(sign-1u64);
        if value!=value {key=full;}
        else if key&sign!=0 {key^=full;}
        else {key^=sign;}
    } else if comptime!(code==3) {key=u64::cast_from(value!=E::cast_from(0));}
    else if comptime!(code==4) {key=u64::cast_from(value)^(1u64<<63u32);}
    else if comptime!(code==5) {key=u64::cast_from(u32::cast_from(value)^0x80000000u32);}
    else if comptime!(code==6) {key=u64::cast_from(u16::cast_from(value)^0x8000u16);}
    else if comptime!(code==7) {key=u64::cast_from(u8::cast_from(value)^0x80u8);}
    else {key=u64::cast_from(value);}
    key
}

#[ruda(launch)]
fn select<E:Numeric+RudaElement>(input:&Tensor<E>,values:&mut Tensor<E>,indices:&mut Tensor<i64>,k:u32,#[comptime] kind:u32,#[comptime] code:u32,#[comptime] bits:u32) {
    let row=(RUDA_POS_X+RUDA_POS_Y*RUDA_COUNT_X) as usize;let lane=UNIT_POS as usize;
    if row<input.shape(0) {
        let width=input.shape(1);let length=i64::cast_from(width);let base=row*width;
        let mut counts=SharedMemory::<u32>::new(1024usize);
        let mut positions=SharedMemory::<u32>::new(256usize);
        let mut selected=SharedMemory::<E>::new(1usize);
        let mut target=k;
        if comptime!(kind!=0) {
            let mut nan_count=0u32;let mut position=i64::cast_from(lane);
            while position<length {
                let value=input[address(input,base+usize::cast_from(position))];
                if value!=value {nan_count+=1;}
                position+=256;
            }
            positions[lane]=nan_count;sync_ruda();let mut step=128usize;
            while step>0 {
                if lane<step {positions[lane]+=positions[lane+step];}
                sync_ruda();step/=2;
            }
            let missing=positions[0];let finite=u32::cast_from(width)-missing;
            target=1;
            if finite>0 {target=(finite-1)/2+1;}
            if comptime!(kind==1) {if missing>0 {target=u32::cast_from(width);}}
            sync_ruda();
        }
        let mut mask=0u64;let mut prefix=0u64;let mut shift=u64::cast_from(bits.runtime());
        while shift>0 {
            shift-=2;
            let mut local=Array::<u32>::new(4usize);
            for bucket in 0..4usize {local[bucket]=0;}
            let mut position=i64::cast_from(lane);
            while position<length {
                let key=radix_key::<E>(input[address(input,base+usize::cast_from(position))],code,bits);
                if key&mask==prefix {local[usize::cast_from((key>>shift)&3u64)]+=1;}
                position+=256;
            }
            for bucket in 0..4usize {counts[bucket*256+lane]=local[bucket];}
            sync_ruda();let mut step=128usize;
            while step>0 {
                if lane<step {
                    for bucket in 0..4usize {counts[bucket*256+lane]+=counts[bucket*256+lane+step];}
                }
                sync_ruda();step/=2;
            }
            let mut bucket=0usize;
            while bucket<3 && target>counts[bucket*256] {target-=counts[bucket*256];bucket+=1;}
            prefix|=u64::cast_from(bucket)<<shift;mask|=3u64<<shift;
            let population=counts[bucket*256];sync_ruda();
            if population==1 {break;}
        }
        let mut first=u32::MAX.runtime();let mut position=i64::cast_from(lane);
        while position<length {
            let key=radix_key::<E>(input[address(input,base+usize::cast_from(position))],code,bits);
            if key&mask==prefix {first=u32::min(first,u32::cast_from(position));}
            position+=256;
        }
        positions[lane]=first;sync_ruda();let mut step=128usize;
        while step>0 {
            if lane<step {positions[lane]=u32::min(positions[lane],positions[lane+step]);}
            sync_ruda();step/=2;
        }
        if lane==0 {
            selected[0]=input[address(input,base+positions[0] as usize)];
            if comptime!(code==3) {selected[0]=E::cast_from(selected[0]!=E::cast_from(0));}
        }
        sync_ruda();let chosen=selected[0];first=u32::MAX.runtime();position=i64::cast_from(lane);
        while position<length {
            let mut value=input[address(input,base+usize::cast_from(position))];
            if comptime!(code==3) {value=E::cast_from(value!=E::cast_from(0));}
            if value==chosen || (value!=value && chosen!=chosen) {first=u32::min(first,u32::cast_from(position));}
            position+=256;
        }
        positions[lane]=first;sync_ruda();step=128;
        while step>0 {
            if lane<step {positions[lane]=u32::min(positions[lane],positions[lane+step]);}
            sync_ruda();step/=2;
        }
        if lane==0 {values[address(values,row)]=chosen;indices[address(indices,row)]=i64::cast_from(positions[0]);}
    }
}

#[ruda(launch)]
fn mode<E:Numeric+RudaElement>(sorted:&Tensor<E>,order:&Tensor<i64>,values:&mut Tensor<E>,indices:&mut Tensor<i64>,#[comptime] last:bool) {
    let row=(RUDA_POS_X+RUDA_POS_Y*RUDA_COUNT_X) as usize;let lane=UNIT_POS as usize;
    if row<sorted.shape(0) {
        let width=sorted.shape(1);let base=row*width;let length=i64::cast_from(width);
        let mut counts=SharedMemory::<u32>::new(256usize);let mut starts=SharedMemory::<u32>::new(256usize);
        let mut best_count=0u32;let mut best_start=u32::MAX.runtime();let mut position=i64::cast_from(lane);
        while position<length {
            let start=usize::cast_from(position);let value=sorted[address(sorted,base+start)];let mut begins=true;
            if start>0 {begins=sorted[address(sorted,base+start-1)]!=value;}
            if begins {
                let mut end=start+1;
                if value==value {
                    let mut low=end;let mut high=width;
                    while low<high {
                        let middle=low+(high-low)/2;
                        if sorted[address(sorted,base+middle)]==value {low=middle+1;}else{high=middle;}
                    }
                    end=low;
                }
                let count=u32::cast_from(end-start);
                if count>best_count || (count==best_count && u32::cast_from(start)<best_start) {
                    best_count=count;best_start=u32::cast_from(start);
                }
            }
            position+=256;
        }
        counts[lane]=best_count;starts[lane]=best_start;sync_ruda();let mut step=128usize;
        while step>0 {
            if lane<step {
                let right=counts[lane+step];let start=starts[lane+step];
                if right>counts[lane] || (right==counts[lane] && start<starts[lane]) {counts[lane]=right;starts[lane]=start;}
            }
            sync_ruda();step/=2;
        }
        if lane==0 {
            let start=starts[0] as usize;let mut selected=start;
            if comptime!(last) {selected+=counts[0] as usize-1;}
            values[address(values,row)]=sorted[address(sorted,base+start)];
            indices[address(indices,row)]=order[address(order,base+selected)];
        }
    }
}

pub(super) fn dispatch(op:u32,xs:&[View],ints:&[i64]) {
    assert!(matches!(op,354|355));assert_eq!(xs.len(),if op==354 {3}else{4});
    let source=&xs[0];let offset=if op==354 {1}else{2};let values=&xs[offset];let indices=&xs[offset+1];
    assert_eq!(source.shape.len(),2);assert_eq!(values.shape.len(),1);assert_eq!(indices.shape,values.shape);
    assert_eq!(values.len,source.shape[0]);assert_eq!(values.dtype,source.dtype);assert_eq!(indices.dtype,4);
    assert!(source.shape[1]>0);assert!(source.dtype<=9);
    let mut kind=0u32;let mut k=1u32;let mut last=false;
    if op==354 {
        assert_eq!(ints.len(),2);kind=u32::try_from(ints[0]).expect("invalid selection kind");assert!(kind<=2);
        k=u32::try_from(ints[1]).expect("selection k outside U32");assert!(k>=1 && k as usize<=source.shape[1]);
    } else {
        assert_eq!(ints.len(),1);assert!(matches!(ints[0],0|1));last=ints[0]!=0;
        assert_eq!(xs[1].shape,source.shape);assert_eq!(xs[1].dtype,4);
    }
    if values.len==0 {return;}
    let code=source.dtype;let bits=[32u32,16,16,8,64,32,16,8,8,64][code as usize];
    let rows=u32::try_from(values.len).expect("selection row count overflow");let x=rows.min(1<<20);
    let grid=RudaCount::Static(x,rows.div_ceil(x),1);let block=RudaDim::new_1d(256);let c=client();
    macro_rules! run {($e:ty)=>{unsafe {
        if op==354 {select::launch::<$e,CudaRuntime>(&c,grid,block,source.arg(),values.arg(),indices.arg(),k,kind,code,bits);}
        else {mode::launch::<$e,CudaRuntime>(&c,grid,block,source.arg(),xs[1].arg(),values.arg(),indices.arg(),last);}
    }};}
    match code {0=>run!(f32),1=>run!(f16),2=>run!(bf16),3|8=>run!(u8),4=>run!(i64),5=>run!(i32),6=>run!(i16),7=>run!(i8),9=>run!(f64),_=>unreachable!()}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
