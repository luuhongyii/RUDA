use super::*;
use super::float8::{left,right};
use ruda_core::{ir::{ElemType,FloatKind,features::MmaConfig},tensor::DType};
use ruda_kernel::quantization::{float8::encode_scaled,packed::decode_code};

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_indexed_block_float8_expert_api_version()->u32 {1}

struct Layout { rows:usize,cols:usize,origin_row:usize,origin_col:usize,block_rows:usize,dynamic:u32,capacity:usize }

fn layouts(m:usize,n:usize,k:usize,ints:&[i64])->[Layout;2] {
    let mode=ints[2];let count=ints[1] as usize;
    let (ra,rb,bk)=(ints[3] as usize,ints[4] as usize,ints[5] as usize);
    let (axis,rank,world)=(ints[6],ints[7] as usize,ints[8] as usize);
    let (ar,br,cols)=if mode==0 {(m,n,k)}else if mode==1 {(m,k,n)}else{(n,k,m)};
    let mut a=Layout {rows:ar,cols,origin_row:0,origin_col:0,block_rows:ra,dynamic:if mode==2 {1}else{0},capacity:0};
    let mut b=Layout {rows:br,cols,origin_row:0,origin_col:0,block_rows:rb,dynamic:if mode==2 {1}else{2},capacity:0};
    if mode==0 && axis==1 || mode==1 && axis==0 {
        a.origin_col=cols.checked_mul(rank).expect("FP8 shard origin overflow");b.origin_col=a.origin_col;
        a.cols=cols.checked_mul(world).expect("FP8 logical contraction overflow");b.cols=a.cols;
    }else if mode==0 && axis==0 || mode==1 && axis==1 || mode==2 && axis==1 {
        b.origin_row=br.checked_mul(rank).expect("FP8 shard origin overflow");
        b.rows=br.checked_mul(world).expect("FP8 logical weight rows overflow");
    }else if mode==2 && axis==0 {
        a.origin_row=ar.checked_mul(rank).expect("FP8 shard origin overflow");
        a.rows=ar.checked_mul(world).expect("FP8 logical gradient rows overflow");
    }
    for p in [&mut a,&mut b] {
        assert!(p.rows<=u32::MAX as usize && p.cols<=u32::MAX as usize);
        let (rows,groups)=match p.dynamic {
            0=>(m.div_ceil(p.block_rows)+count-1,p.cols.div_ceil(bk)),
            1=>(p.rows.div_ceil(p.block_rows),m.div_ceil(bk)+count-1),
            _=>(p.rows.div_ceil(p.block_rows).checked_mul(count).expect("FP8 block rows overflow"),p.cols.div_ceil(bk)),
        };
        p.capacity=rows.checked_mul(groups).expect("FP8 statistics size overflow");
        assert!(p.capacity<=u32::MAX as usize/2);
    }
    [a,b]
}

pub(crate) fn validate(vs:&[View],ints:&[i64],product:bool) {
    assert_eq!(ints.len(),if product {15}else{9});
    assert!((1..=16).contains(&ints[1]) && ints[0]>=0 && ints[0]<=u32::MAX as i64-ints[1]);
    let count=ints[1] as usize;let mode=ints[2];assert!(matches!(mode,0|1|2));
    assert!(ints[3..6].iter().all(|&v|v>0 && v<=u32::MAX as i64) && ints[5]%32==0);
    for rows in &ints[3..5] {assert!(rows.checked_mul(ints[5]).is_some_and(|n|n<=u32::MAX as i64-32));}
    assert!(matches!(ints[6],-1|0|1) && ints[8]>0 && ints[8]<=u32::MAX as i64 && ints[7]>=0 && ints[7]<ints[8]);
    if ints[6]==-1 {assert_eq!((ints[7],ints[8]),(0,1));}
    assert_eq!(vs.len(),if product {count+5+if mode==2 {count}else{1}}else{count+5});
    for value in vs {crate::operation_graph::packed(value);}
    let (x,ids,w,g)=(&vs[0],&vs[1],&vs[2],&vs[count+2]);
    assert!(x.shape.len()==2 && x.dtype<=2 && x.shape[1]>0 && matches!(ids.dtype,4|5));
    assert_eq!(ids.shape,vec![x.shape[0]]);assert!(w.shape.len()==2 && w.shape.iter().all(|&n|n>0));
    for weight in &vs[2..2+count] {assert_eq!(weight.shape,w.shape);assert!(weight.dtype<=2);}
    assert_eq!(x.shape[1],w.shape[if mode==1 {0}else{1}]);
    if mode==2 {assert_eq!(g.shape,vec![x.shape[0],w.shape[0]]);assert!(g.dtype<=2);}
    else {assert_eq!(g.shape,vec![0]);assert_eq!(g.dtype,x.dtype);}
    let layout=layouts(x.shape[0],w.shape[0],w.shape[1],ints);
    for (role,p) in layout.iter().enumerate() {
        let stats=&vs[count+3+role];assert_eq!(stats.dtype,0);assert_eq!(stats.shape,vec![2,p.capacity]);
    }
    if product {
        assert!(matches!(ints[9],18|19) && matches!(ints[10],18|19));
        assert!((0..=16).contains(&ints[11]) && matches!(ints[12],0|1) && (0..=2).contains(&ints[13]));
        assert!(ints[14]>=0 && ints[14]<(1i64<<count));
        for (e,out) in vs[count+5..].iter().enumerate() {
            assert_eq!(out.dtype,0);
            assert_eq!(out.shape,if mode==2 {if ints[14]&(1<<e)!=0 {w.shape.clone()}else{vec![0]}}
                else {vec![x.shape[0],w.shape[if mode==0 {0}else{1}]]});
        }
    }
}

pub(crate) fn prepare(c:&ComputeClient<CudaRuntime>,vs:&[View],ints:&[i64],product:bool)->GroupedKernelPlan<CudaRuntime> {
    validate(vs,ints,product);
    let ts:Vec<_>=vs.iter().map(|v|{let mut t=primitives::tensor(v);t.client=c.clone();t}).collect();
    let count=ints[1] as usize;let (m,n,k)=(vs[0].shape[0],vs[2].shape[0],vs[2].shape[1]);let mode=ints[2] as u32;
    let layout=layouts(m,n,k,ints);let a_layout=&layout[0];let b_layout=&layout[1];
    let h=&c.properties().hardware;assert!(h.plane_size_min==32 && h.plane_size_max==32);
    let mut rows=tensorcore::prepare_rows(c,&ts[1],ints[0] as u32,count);
    let zeros:Vec<usize>=if product {if mode==2 {Vec::new()}else{vec![count+5]}}else{vec![count+3,count+4]};
    for index in zeros {
        if vs[index].len!=0 {rows.plan.kernels.push(unsafe {tensorcore::zero::prepare::<CudaRuntime>(c,
            RudaCount::Static(vs[index].len.div_ceil(128) as u32,1,1),RudaDim::new_1d(128),ts[index].clone().into_array_arg(),DType::F32.into())});}
    }
    let ty=|format|ElemType::Float(if format==18 {FloatKind::E4M3}else{FloatKind::E5M2}).into();
    let native=product && ints[13]!=2 && c.features().matmul.mma.contains(&MmaConfig {
        a_type:ty(ints[9]),b_type:ty(ints[10]),cd_type:DType::F32.into(),m:16,n:8,k:32});
    if product {
        assert!(ints[13]!=1 || native,"native indexed block FP8 MMA is unavailable");
        assert!(native || h.max_shared_memory_size>=2048 && c.features().matmul.cmma.contains(&MmaConfig {
            a_type:DType::F16.into(),b_type:DType::F16.into(),cd_type:DType::F32.into(),m:16,n:16,k:16}),
            "indexed block FP8 needs native MMA or exact FP16 carriers");
    }
    for e in 0..count {
        let a=&ts[if mode==2 {count+2}else{0}];let b=&ts[if mode==2 {0}else{2+e}];
        if !product {
            for (role,p) in layout.iter().enumerate() {
                if p.capacity==0 {continue;}
                let value=if role==0 {a}else{b};let blocks=p.capacity.min(4096).min(h.max_ruda_count.0 as usize);assert!(blocks>0);
                rows.plan.kernels.push(unsafe {block_statistics::prepare::<CudaRuntime>(c,
                    RudaCount::Static(blocks as u32,1,1),RudaDim::new_1d(32),value.clone().into_array_arg(),
                    rows.indices.clone().into_array_arg(),rows.offsets.clone().into_array_arg(),ts[count+3+role].clone().into_array_arg(),
                    e as u32,k as u32,n as u32,p.origin_row as u32,p.origin_col as u32,p.rows as u32,p.cols as u32,
                    blocks as u32,p.block_rows as u32,ints[5] as u32,p.dynamic,mode,role==0,value.dtype.into())});
            }
            continue;
        }
        let out=&ts[count+5+if mode==2 {e}else{0}];if out.meta.num_elements()==0 {continue;}
        let width=if mode==0 {n}else{k};let height=if mode==2 {n}else{m};
        let blocks=height.div_ceil(16).min(128).min(h.max_ruda_count.1 as usize);assert!(blocks>0);
        let tile=if native {8}else{16};assert!(width.div_ceil(tile)<=h.max_ruda_count.0 as usize);
        let grid=RudaCount::Static(width.div_ceil(tile) as u32,blocks as u32,1);
        let maximum=|format|if format==18 {448.0f32}else{57344.0f32};
        let maximum_a=maximum(ints[9])*2.0f32.powi(-(ints[11] as i32));let maximum_b=maximum(ints[10])*2.0f32.powi(-(ints[11] as i32));
        unsafe {
            let node=if native {native_product::prepare::<CudaRuntime>(c,grid,RudaDim::new_1d(32),
                a.clone().into_array_arg(),b.clone().into_array_arg(),rows.indices.clone().into_array_arg(),rows.offsets.clone().into_array_arg(),
                ts[count+3].clone().into_array_arg(),ts[count+4].clone().into_array_arg(),out.clone().into_array_arg(),
                e as u32,k as u32,n as u32,(blocks*16) as u32,a_layout.origin_row as u32,b_layout.origin_row as u32,a_layout.origin_col as u32,
                a_layout.rows as u32,b_layout.rows as u32,a_layout.cols as u32,maximum_a,maximum_b,
                ints[3] as u32,ints[4] as u32,ints[5] as u32,mode,ints[9] as u32,ints[10] as u32,ints[12]!=0,
                a.dtype.into(),b.dtype.into(),ty(ints[9]),ty(ints[10]))}
            else {carrier_product::prepare::<CudaRuntime>(c,grid,RudaDim::new_1d(32),
                a.clone().into_array_arg(),b.clone().into_array_arg(),rows.indices.clone().into_array_arg(),rows.offsets.clone().into_array_arg(),
                ts[count+3].clone().into_array_arg(),ts[count+4].clone().into_array_arg(),out.clone().into_array_arg(),
                e as u32,k as u32,n as u32,(blocks*16) as u32,a_layout.origin_row as u32,b_layout.origin_row as u32,a_layout.origin_col as u32,
                a_layout.rows as u32,b_layout.rows as u32,a_layout.cols as u32,maximum_a,maximum_b,
                ints[3] as u32,ints[4] as u32,ints[5] as u32,mode,ints[9] as u32,ints[10] as u32,ints[12]!=0,
                a.dtype.into(),b.dtype.into())};
            rows.plan.kernels.push(node);
        }
    }
    rows.plan
}

#[ruda]
fn block_info(offsets:&Array<u32>,expert:u32,global_rows:u32,global_cols:u32,
    #[comptime] rows:u32,#[comptime] cols:u32,#[comptime] dynamic:u32)->(usize,usize,usize,usize) {
    let mut previous=0u32;let mut base=0usize;
    while previous<expert {
        let count=offsets[previous as usize+1]-offsets[previous as usize];
        let r=if comptime!(dynamic==0) {count}else{global_rows};let c=if comptime!(dynamic==1) {count}else{global_cols};
        base+=(r/rows+u32::cast_from(r%rows!=0)) as usize*(c/cols+u32::cast_from(c%cols!=0)) as usize;previous+=1;
    }
    let count=offsets[expert as usize+1]-offsets[expert as usize];let r=if comptime!(dynamic==0) {count}else{global_rows};
    let c=if comptime!(dynamic==1) {count}else{global_cols};
    (base,r as usize,c as usize,(c/cols+u32::cast_from(c%cols!=0)) as usize)
}

#[ruda(launch)]
fn block_statistics<F:Float>(value:&Array<F>,indices:&Array<u32>,offsets:&Array<u32>,out:&mut Array<f32>,
    expert:u32,inner:u32,columns:u32,origin_row:u32,origin_col:u32,global_rows:u32,global_cols:u32,step:u32,
    #[comptime] block_rows:u32,#[comptime] block_cols:u32,#[comptime] dynamic:u32,#[comptime] mode:u32,
    #[comptime] operand_a:bool,#[define(F)] _dtype:StorageType) {
    let begin=offsets[expert as usize] as usize;let count=(offsets[expert as usize+1]-offsets[expert as usize]) as usize;
    let (base,height,_,groups)=block_info(offsets,expert,global_rows,global_cols,block_rows,block_cols,dynamic);
    let row_blocks=height/block_rows as usize+usize::cast_from(height%block_rows as usize!=0);
    let total=row_blocks*groups;let mut block=RUDA_POS_X as usize;let capacity=out.len()/2;
    let m=if comptime!(mode==2) {columns as usize}else{count};let n=if comptime!(mode==0) {columns as usize}else{inner as usize};
    let k=if comptime!(mode==2) {count}else if comptime!(mode==0) {inner as usize}else{columns as usize};
    let local_rows=if comptime!(operand_a) {m}else{n};
    while block<total {
        let row_start=block/groups*block_rows as usize;let col_start=block%groups*block_cols as usize;
        let mut maximum=0.0f32;let mut invalid=0.0f32;let mut p=UNIT_POS;
        while p<block_rows*block_cols {
            let row=row_start+(p/block_cols) as usize;let col=col_start+(p%block_cols) as usize;
            if count!=0 && row>=origin_row as usize && col>=origin_col as usize {
                let r=row-origin_row as usize;let c=col-origin_col as usize;
                if r<local_rows && c<k {
                    let v=if comptime!(operand_a) {left(value,indices,begin,r,c,inner as usize,columns as usize,mode)}
                        else {right(value,indices,begin,r,c,inner as usize,mode)};
                    let magnitude=v.abs();
                    if magnitude.to_bits()>=0x7f800000u32 {invalid=1.0;}else {maximum=f32::max(maximum,magnitude);}
                }
            }
            p+=32;
        }
        maximum=plane_max(maximum);invalid=plane_max(invalid);
        if UNIT_POS==0 {out[base+block]=maximum;out[capacity+base+block]=invalid;}
        if total-block<=step as usize {block=total;}else {block+=step as usize;}
    }
}

#[ruda]
fn scale(statistics:&Array<f32>,base:usize,row:usize,group:usize,groups:usize,maximum:f32,
    #[comptime] block_rows:u32,#[comptime] power:bool)->f32 {
    let index=base+(row/block_rows as usize)*groups+group;
    let observed=statistics[index];let invalid=statistics[statistics.len()/2+index];let mut result=1.0f32;
    if invalid==0.0 && observed>0.0 {result=f32::max(observed/maximum,f32::from_bits(0x00800000u32));}
    if comptime!(power) {
        let bits=result.to_bits();let mut exponent=bits&0x7f800000u32;
        if bits&0x007fffffu32!=0u32 {exponent+=0x00800000u32;}result=f32::from_bits(exponent);
    }
    result
}

#[ruda(launch)]
fn carrier_product<L:Float,R:Float>(a:&Array<L>,b:&Array<R>,indices:&Array<u32>,offsets:&Array<u32>,sa:&Array<f32>,sb:&Array<f32>,out:&mut Array<f32>,
    expert:u32,inner:u32,columns:u32,step:u32,origin_a:u32,origin_b:u32,origin_k:u32,global_a:u32,global_b:u32,global_k:u32,
    maximum_a:f32,maximum_b:f32,#[comptime] bra:u32,#[comptime] brb:u32,#[comptime] bk:u32,#[comptime] mode:u32,
    #[comptime] format_a:u32,#[comptime] format_b:u32,#[comptime] power:bool,#[define(L)] _left:StorageType,#[define(R)] _right:StorageType) {
    let begin=offsets[expert as usize] as usize;let count=(offsets[expert as usize+1]-offsets[expert as usize]) as usize;
    let (base_a,_,_,groups)=block_info(offsets,expert,global_a,global_k,bra,bk,comptime!(if mode==2 {1u32}else{0u32}));
    let (base_b,_,_,_)=block_info(offsets,expert,global_b,global_k,brb,bk,comptime!(if mode==2 {1u32}else{2u32}));
    let m=if comptime!(mode==2) {columns as usize}else{count};let n=if comptime!(mode==0) {columns as usize}else{inner as usize};
    let k=if comptime!(mode==2) {count}else if comptime!(mode==0) {inner as usize}else{columns as usize};
    let end_k=origin_k as usize+k;let last_group=end_k/bk as usize+usize::cast_from(end_k%bk as usize!=0);
    let lane=UNIT_POS as usize;let col=RUDA_POS_X as usize*16;let mut row=RUDA_POS_Y as usize*16;
    let mut al=SharedMemory::<f16>::new_aligned(256usize,32usize);let mut bl=SharedMemory::<f16>::new_aligned(256usize,32usize);
    let mut result=SharedMemory::<f32>::new_aligned(256usize,32usize);let mut sum=Array::<f32>::new(8usize);
    while row<m {
        #[unroll] for i in 0usize..8usize {sum[i]=0.0;}
        let mut group=origin_k as usize/bk as usize;
        while group<last_group {
            let acc=cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,16usize,16usize,16usize,cmma::MatrixLayout::Undefined,0.0f32);
            let mut base=0usize;
            while base<bk as usize {
                #[unroll] for i in 0usize..8usize {
                    let t=lane+i*32;let r=t/16;let c=t%16;
                    let ak=group*bk as usize+base+c;let bkc=group*bk as usize+base+r;
                    let mut av=0.0f32;let mut bv=0.0f32;
                    if row+r<m && ak>=origin_k as usize && ak<end_k {
                        let s=scale(sa,base_a,origin_a as usize+row+r,group,groups,maximum_a,bra,power);
                        av=decode_code(encode_scaled(left(a,indices,begin,row+r,ak-origin_k as usize,inner as usize,columns as usize,mode),s,format_a) as u32,format_a);
                    }
                    if col+c<n && bkc>=origin_k as usize && bkc<end_k {
                        let s=scale(sb,base_b,origin_b as usize+col+c,group,groups,maximum_b,brb,power);
                        bv=decode_code(encode_scaled(right(b,indices,begin,col+c,bkc-origin_k as usize,inner as usize,mode),s,format_b) as u32,format_b);
                    }
                    al[t]=f16::cast_from(av);bl[t]=f16::cast_from(bv);
                }
                sync_ruda();
                let aa=cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::A,16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&al.to_slice(),16);
                let bb=cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B,16usize,16usize,16usize,cmma::MatrixLayout::RowMajor,&bl.to_slice(),16);
                cmma::execute::<f16,f16,f32,f32, _>(&aa,&bb,&acc,&acc);sync_ruda();base+=16;
            }
            cmma::store(&mut result.to_slice_mut(),&acc,16,cmma::MatrixLayout::RowMajor);sync_ruda();
            #[unroll] for i in 0usize..8usize {
                let t=lane+i*32;let rr=row+t/16;let cc=col+t%16;
                if rr<m && cc<n {
                    let ascale=scale(sa,base_a,origin_a as usize+rr,group,groups,maximum_a,bra,power);
                    let bscale=scale(sb,base_b,origin_b as usize+cc,group,groups,maximum_b,brb,power);
                    sum[i]+=result[t]*ascale*bscale;
                }
            }
            sync_ruda();group+=1;
        }
        #[unroll] for i in 0usize..8usize {
            let t=lane+i*32;let rr=row+t/16;let cc=col+t%16;
            if rr<m && cc<n {let destination=if comptime!(mode==2) {rr}else{indices[begin+rr] as usize};out[destination*n+cc]=sum[i];}
        }
        if m-row<=step as usize {row=m;}else {row+=step as usize;}
    }
}

#[ruda(launch)]
fn native_product<L:Float,R:Float,A:Float,B:Float>(a:&Array<L>,b:&Array<R>,indices:&Array<u32>,offsets:&Array<u32>,sa:&Array<f32>,sb:&Array<f32>,out:&mut Array<f32>,
    expert:u32,inner:u32,columns:u32,step:u32,origin_a:u32,origin_b:u32,origin_k:u32,global_a:u32,global_b:u32,global_k:u32,
    maximum_a:f32,maximum_b:f32,#[comptime] bra:u32,#[comptime] brb:u32,#[comptime] bk:u32,#[comptime] mode:u32,
    #[comptime] format_a:u32,#[comptime] format_b:u32,#[comptime] power:bool,#[define(L)] _left:StorageType,#[define(R)] _right:StorageType,
    #[define(A)] _a_type:StorageType,#[define(B)] _b_type:StorageType) {
    let begin=offsets[expert as usize] as usize;let count=(offsets[expert as usize+1]-offsets[expert as usize]) as usize;
    let (base_a,_,_,groups)=block_info(offsets,expert,global_a,global_k,bra,bk,comptime!(if mode==2 {1u32}else{0u32}));
    let (base_b,_,_,_)=block_info(offsets,expert,global_b,global_k,brb,bk,comptime!(if mode==2 {1u32}else{2u32}));
    let m=if comptime!(mode==2) {columns as usize}else{count};let n=if comptime!(mode==0) {columns as usize}else{inner as usize};
    let k=if comptime!(mode==2) {count}else if comptime!(mode==0) {inner as usize}else{columns as usize};
    let end_k=origin_k as usize+k;let last_group=end_k/bk as usize+usize::cast_from(end_k%bk as usize!=0);
    let lane=UNIT_POS_PLANE;let col=RUDA_POS_X as usize*8;let mut row=RUDA_POS_Y as usize*16;
    let def=cmma::MmaDefinition::<A,B,f32>::new(16usize,8usize,32usize);
    let va=def.vector_size(cmma::MatrixIdent::A);let vb=def.vector_size(cmma::MatrixIdent::B);let vc=def.vector_size(cmma::MatrixIdent::Accumulator);
    let size!(NA)=va;let size!(NB)=vb;let size!(NC)=vc;
    let ca=def.vectors_per_lane(cmma::MatrixIdent::A);let cb=def.vectors_per_lane(cmma::MatrixIdent::B);let cc=def.vectors_per_lane(cmma::MatrixIdent::Accumulator);
    let mut ra=Array::<Vector<A,NA>>::new(ca);let mut rb=Array::<Vector<B,NB>>::new(cb);
    let mut rc=Array::<Vector<f32,NC>>::new(cc);let mut sum=Array::<Vector<f32,NC>>::new(cc);
    while row<m {
        #[unroll] for i in 0..cc {sum[i]=Vector::cast_from(0.0f32);}
        let mut group=origin_k as usize/bk as usize;
        while group<last_group {
            #[unroll] for i in 0..cc {rc[i]=Vector::cast_from(0.0f32);}
            let mut base=0usize;
            while base<bk as usize {
                #[unroll] for i in 0..ca {
                    let mut reg=Vector::empty();
                    #[unroll] for j in 0..va {
                        let (r,c)=def.position_of_nth(lane,(i*va+j) as u32,cmma::MatrixIdent::A);
                        let rr=row+r as usize;let kk=group*bk as usize+base+c as usize;let mut code=0u8;
                        if rr<m && kk>=origin_k as usize && kk<end_k {
                            let s=scale(sa,base_a,origin_a as usize+rr,group,groups,maximum_a,bra,power);
                            code=encode_scaled(left(a,indices,begin,rr,kk-origin_k as usize,inner as usize,columns as usize,mode),s,format_a);
                        }
                        reg[j]=A::reinterpret(code);
                    }
                    ra[i]=reg;
                }
                #[unroll] for i in 0..cb {
                    let mut reg=Vector::empty();
                    #[unroll] for j in 0..vb {
                        let (r,c)=def.position_of_nth(lane,(i*vb+j) as u32,cmma::MatrixIdent::B);
                        let column=col+c as usize;let kk=group*bk as usize+base+r as usize;let mut code=0u8;
                        if column<n && kk>=origin_k as usize && kk<end_k {
                            let s=scale(sb,base_b,origin_b as usize+column,group,groups,maximum_b,brb,power);
                            code=encode_scaled(right(b,indices,begin,column,kk-origin_k as usize,inner as usize,mode),s,format_b);
                        }
                        reg[j]=B::reinterpret(code);
                    }
                    rb[i]=reg;
                }
                def.execute_inplace(&ra,&rb,&mut rc);base+=32;
            }
            #[unroll] for i in 0..cc {
                let reg=rc[i];let mut total=sum[i];
                #[unroll] for j in 0..vc {
                    let (r,c)=def.position_of_nth(lane,(i*vc+j) as u32,cmma::MatrixIdent::Accumulator);let rr=row+r as usize;let column=col+c as usize;
                    if rr<m && column<n {
                        let ascale=scale(sa,base_a,origin_a as usize+rr,group,groups,maximum_a,bra,power);
                        let bscale=scale(sb,base_b,origin_b as usize+column,group,groups,maximum_b,brb,power);
                        total[j]+=reg[j]*ascale*bscale;
                    }
                }
                sum[i]=total;
            }
            group+=1;
        }
        #[unroll] for i in 0..cc {
            let reg=sum[i];
            #[unroll] for j in 0..vc {
                let (r,c)=def.position_of_nth(lane,(i*vc+j) as u32,cmma::MatrixIdent::Accumulator);let rr=row+r as usize;let column=col+c as usize;
                if rr<m && column<n {let destination=if comptime!(mode==2) {rr}else{indices[begin+rr] as usize};out[destination*n+column]=reg[j];}
            }
        }
        if m-row<=step as usize {row=m;}else {row+=step as usize;}
    }
}
