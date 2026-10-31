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
    if ai==u32::MAX {order=false;} else if bi==u32::MAX {order=true;}
    order
}

#[ruda]
fn ordered_key<E:Numeric,U:Int>(value:E,#[comptime] dtype:u32,#[comptime] bits:u32,
    #[comptime] descending:bool,#[comptime] equal_zero:bool)->U {
    let limit=U::cast_from(comptime!(u64::MAX>>(64-bits)));
    let sign=U::cast_from(comptime!(1u64<<(bits-1)));
    let mut key=U::cast_from(0u32);
    if comptime!(dtype<=2 || dtype==9) {
        if comptime!(dtype==9) {key=U::cast_from(u64::reinterpret(value));}
        else if comptime!(dtype==0) {key=U::cast_from(u32::reinterpret(value));}
        else {key=U::cast_from(u16::reinterpret(value));}
        if comptime!(equal_zero) {if value==E::cast_from(0u32) {key=U::cast_from(0u32);}}
        if (key&sign)!=U::cast_from(0u32) {key^=limit;}else {key^=sign;}
        if value!=value {key=limit;}
    } else {
        key=U::cast_from(value)&limit;
        if comptime!(dtype!=8 && dtype!=3) {key^=sign;}
    }
    if comptime!(descending) {key^=limit;}
    key
}

#[ruda]
fn topk_before<E:Numeric>(a:E,ai:u32,b:E,bi:u32,#[comptime] dtype:u32,#[comptime] bits:u32,#[comptime] descending:bool)->bool {
    let ak=ordered_key::<E,u64>(a,dtype,bits,descending,false);
    let bk=ordered_key::<E,u64>(b,dtype,bits,descending,false);
    let mut order=ak<bk || ak==bk && ai<bi;
    if ai==u32::MAX {order=false;}else if bi==u32::MAX {order=true;}
    order
}

#[ruda(launch)]
fn warp_sort<E:Numeric+RudaElement,I:Int+RudaElement>(input:&Tensor<E>,out:&mut Tensor<E>,indices:&mut Tensor<I>,
    rows:usize,inner:usize,#[comptime] length:usize,#[comptime] k:usize,
    #[comptime] chunks:usize,#[comptime] levels:usize,#[comptime] rank:usize,
    #[comptime] descending:bool,#[comptime] indices_only:bool,#[comptime] boolean:bool,#[comptime] runs:bool) {
    let task=RUDA_POS_X as usize*4usize+UNIT_POS as usize/32usize;
    let mut row=task;let mut base=0usize;
    if comptime!(runs) {row=task/comptime!(length.div_ceil(256));base=task%comptime!(length.div_ceil(256))*256usize;}
    let lane=UNIT_POS as usize%32usize;
    if task<rows {
        let mut values=Array::<E>::new(chunks);let mut keys=Array::<u32>::new(chunks);
        #[unroll] for item in 0usize..chunks {
            let column=base+item*32usize+lane;
            let mut value=E::cast_from(0);let mut key=u32::MAX.runtime();
            if column<length {
                value=input[row/inner*length*inner+column*inner+row%inner];key=column as u32;
                if comptime!(boolean) {value=E::cast_from(value!=E::cast_from(0));}
            }
            values[item]=value;keys[item]=key;
        }
        #[unroll] for level in 0usize..levels {
            let span=comptime!(1usize<<(level+1));
            #[unroll] for pass in 0usize..comptime!(level+1) {
                let distance=comptime!(1usize<<(level-pass));
                if comptime!(distance>=32) {
                    #[unroll] for item in 0usize..chunks {
                        if comptime!(item & (distance/32) == 0) {
                            let peer=comptime!(item ^ (distance/32));
                            let a=values[item];let ai=keys[item];let b=values[peer];let bi=keys[peer];
                            let forward=(item*32usize+lane)&span==0;
                            let mut swap=before::<E>(b,bi,a,ai,descending);
                            if !forward {swap=before::<E>(a,ai,b,bi,descending);}
                            if swap {values[item]=b;keys[item]=bi;values[peer]=a;keys[peer]=ai;}
                        }
                    }
                } else {
                    #[unroll] for item in 0usize..chunks {
                        let a=values[item];let ai=keys[item];
                        let b=plane_shuffle_xor(a,distance as u32);let bi=plane_shuffle_xor(ai,distance as u32);
                        let position=item*32usize+lane;
                        let earlier=((position&distance)==0)==((position&span)==0);
                        let mut swap=before::<E>(b,bi,a,ai,descending);
                        if !earlier {swap=before::<E>(a,ai,b,bi,descending);}
                        if swap {values[item]=b;keys[item]=bi;}
                    }
                }
            }
        }
        #[unroll] for item in 0usize..chunks {
            let column=item*32usize+lane;
            if comptime!(runs) {
                if base+column<length {out[row*length+base+column]=values[item];indices[row*length+base+column]=I::cast_from(keys[item]);}
            } else if column<k {
                let mut position=row/inner*k*inner+column*inner+row%inner;
                let mut vi=0usize;let mut ii=0usize;
                #[unroll] for axis in 0usize..rank {
                    let dim=comptime!(rank-axis-1);let coordinate=position%indices.shape(dim);
                    position/=indices.shape(dim);ii+=coordinate*indices.stride(dim);
                    if comptime!(!indices_only) {vi+=coordinate*out.stride(dim);}
                }
                indices[ii]=I::cast_from(keys[item]);
                if comptime!(!indices_only) {out[vi]=values[item];}
            }
        }
    }
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
fn warp_topk<E:Numeric+RudaElement>(input:&Tensor<E>,out:&mut Tensor<E>,indices:&mut Tensor<i64>,
    rows:usize,inner:usize,#[comptime] length:usize,#[comptime] chunks:usize,
    #[comptime] k:usize,#[comptime] rank:usize,#[comptime] descending:bool,#[comptime] ordered:bool,
    #[comptime] dtype:u32,#[comptime] bits:u32)
{
    let row=RUDA_POS_X as usize*4usize+UNIT_POS as usize/32usize;
    let lane=UNIT_POS as usize%32usize;
    if row<rows {
        let mut values=Array::<E>::new(chunks);let mut keys=Array::<u32>::new(chunks);
        let mut selected_values=Array::<E>::new(k);let mut selected_keys=Array::<u32>::new(k);
        #[unroll] for item in 0usize..chunks {
            let column=item*32usize+lane;values[item]=E::cast_from(0);keys[item]=u32::MAX;
            if column<length {values[item]=input[row/inner*length*inner+column*inner+row%inner];keys[item]=column as u32;}
        }
        #[unroll] for selected in 0usize..k {
            let mut best=E::cast_from(0);let mut key=u32::MAX.runtime();
            #[unroll] for item in 0usize..chunks {
                if topk_before::<E>(values[item],keys[item],best,key,dtype,bits,descending) {best=values[item];key=keys[item];}
            }
            #[unroll] for pass in 0usize..5 {
                let candidate=plane_shuffle_xor(best,comptime!(1u32<<pass));
                let candidate_key=plane_shuffle_xor(key,comptime!(1u32<<pass));
                if topk_before::<E>(candidate,candidate_key,best,key,dtype,bits,descending) {best=candidate;key=candidate_key;}
            }
            selected_values[selected]=best;selected_keys[selected]=key;
            #[unroll] for item in 0usize..chunks {if keys[item]==key {keys[item]=u32::MAX;}}
        }
        if lane<k {
            let mut best=E::cast_from(0);let mut key=u32::MAX.runtime();let mut destination=lane;
            #[unroll] for selected in 0usize..k {
                if lane==selected {best=selected_values[selected];key=selected_keys[selected];}
            }
            if comptime!(!ordered) {
                let cutoff=selected_values[k-1];
                let strict=topk_before::<E>(best,0u32,cutoff,0u32,dtype,bits,descending);
                destination=0;
                #[unroll] for other in 0usize..k {
                    let other_strict=topk_before::<E>(selected_values[other],0u32,cutoff,0u32,dtype,bits,descending);
                    if other_strict && !strict || other_strict==strict && selected_keys[other]<key {destination+=1;}
                }
            }
            let mut position=row/inner*k*inner+destination*inner+row%inner;
            let mut vi=0usize;let mut ii=0usize;
            #[unroll] for reverse in 0usize..rank {
                let axis=comptime!(rank-1-reverse);let coordinate=position%indices.shape(axis);position/=indices.shape(axis);
                vi+=coordinate*out.stride(axis);ii+=coordinate*indices.stride(axis);
            }
            out[vi]=best;indices[ii]=i64::cast_from(key);
        }
    }
}

#[ruda]
fn write_topk<E:Numeric>(out:&mut Tensor<E>,indices:&mut Tensor<i64>,row:usize,column:usize,source:u32,value:E,
    inner:usize,#[comptime] k:usize,#[comptime] rank:usize) {
    let mut position=row/inner*k*inner+column*inner+row%inner;let mut vi=0usize;let mut ii=0usize;
    #[unroll] for reverse in 0usize..rank {
        let axis=comptime!(rank-1-reverse);let coordinate=position%indices.shape(axis);position/=indices.shape(axis);
        vi+=coordinate*out.stride(axis);ii+=coordinate*indices.stride(axis);
    }
    out[vi]=value;indices[ii]=i64::cast_from(source);
}

#[ruda(launch)]
fn block_topk<E:Numeric+RudaElement,U:Int+RudaElement>(input:&Tensor<E>,out:&mut Tensor<E>,indices:&mut Tensor<i64>,
    inner:usize,#[comptime] length:usize,#[comptime] chunks:usize,#[comptime] k:usize,#[comptime] padded:usize,
    #[comptime] rank:usize,#[comptime] dtype:u32,#[comptime] bits:u32,#[comptime] descending:bool,#[comptime] ordered:bool) {
    let row=RUDA_POS_X as usize;let tid=UNIT_POS as usize;let lane=tid%32;
    let mut values=Array::<E>::new(chunks);let mut keys=Array::<U>::new(chunks);
    #[unroll] for item in 0usize..chunks {
        let column=item*256+tid;let mut value=E::cast_from(0u32);
        if column<length {value=input[row/inner*length*inner+column*inner+row%inner];}
        values[item]=value;keys[item]=ordered_key::<E,U>(value,dtype,bits,descending,false);
    }
    let mut hist=SharedMemory::<u32>::new(32usize);
    let mut prefix=U::cast_from(0u32);let mut mask=U::cast_from(0u32);let mut wanted=k as u32;
    for pass in 0u32..comptime!(bits/2) {
        let shift=U::cast_from(bits-2u32*(pass+1u32));
        let mut counts=Array::<u32>::new(4usize);
        #[unroll] for bin in 0usize..4 {counts[bin]=0u32;}
        #[unroll] for item in 0usize..chunks {
            let key=keys[item];let active=item*256+tid<length && (key&mask)==prefix;
            let digit=(key>>shift)&U::cast_from(3u32);
            #[unroll] for bin in 0usize..4 {counts[bin]+=plane_ballot(active && digit==U::cast_from(bin))[0usize].count_ones();}
        }
        if lane==0 {#[unroll] for bin in 0usize..4 {hist[(tid/32)*4+bin]=counts[bin];}}
        sync_ruda();
        let mut chosen=0u32;let mut found=false;
        #[unroll] for bin in 0usize..4 {
            let mut count=0u32;#[unroll] for warp in 0usize..8 {count+=hist[warp*4+bin];}
            if !found {if wanted>count {wanted-=count;}else {chosen=bin as u32;found=true;}}
        }
        prefix|=U::cast_from(chosen)<<shift;mask|=U::cast_from(3u32)<<shift;sync_ruda();
    }
    let mut selected=SharedMemory::<E>::new(padded);let mut selected_keys=SharedMemory::<u32>::new(padded);
    let mut warps=SharedMemory::<u32>::new(8usize);
    if tid<padded {selected[tid]=E::cast_from(0u32);selected_keys[tid]=u32::MAX;}sync_ruda();
    let mut written=0u32;
    #[unroll] for phase in 0usize..2 {
        #[unroll] for item in 0usize..chunks {
            let column=item*256+tid;let mut take=column<length && keys[item]<prefix;
            if comptime!(phase==1) {take=column<length && keys[item]==prefix;}
            let inclusive=crate::unique_ops::block_prefix(u32::cast_from(take),&mut warps);
            let position=written+inclusive-1u32;let count=warps[7usize];
            if take && position<k as u32 {
                if comptime!(ordered) {selected[position as usize]=values[item];selected_keys[position as usize]=column as u32;}
                else {write_topk(out,indices,row,position as usize,column as u32,values[item],inner,k,rank);}
            }
            written+=count;sync_ruda();
        }
    }
    if comptime!(ordered) {
        #[unroll] for level in 0usize..comptime!(padded.trailing_zeros() as usize) {
            let span=comptime!(1usize<<(level+1));
            #[unroll] for pass in 0usize..comptime!(level+1) {
                let distance=comptime!(1usize<<(level-pass));let mut value=E::cast_from(0u32);let mut key=u32::MAX.runtime();
                if tid<padded {
                    value=selected[tid];key=selected_keys[tid];let peer=tid^distance;
                    let other=selected[peer];let other_key=selected_keys[peer];
                    let earlier=((tid&distance)==0)==((tid&span)==0);
                    let mut exchange=before::<E>(other,other_key,value,key,descending);
                    if !earlier {exchange=before::<E>(value,key,other,other_key,descending);}
                    if exchange {value=other;key=other_key;}
                }
                sync_ruda();if tid<padded {selected[tid]=value;selected_keys[tid]=key;}sync_ruda();
            }
        }
        if tid<k {write_topk(out,indices,row,tid,selected_keys[tid],selected[tid],inner,k,rank);}
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
fn reuse_or_allocate(c:&ComputeClient<CudaRuntime>,shape:&[usize],dtype:u32,reuse:&[View],index:usize)->View {
    if let Some(value)=reuse.get(index) {
        assert_eq!(value.shape,shape);assert_eq!(value.dtype,dtype);
        View::packed(value.handle.clone(),shape.to_vec(),dtype)
    } else {allocate(c,shape,dtype)}
}

#[ruda]
fn radix_rank<E:Numeric>(value:E,valid:bool,histogram:&mut SharedMemory<u32>,
    #[comptime] shift:u32,#[comptime] bits:u32,#[comptime] dtype:u32,#[comptime] descending:bool)->(u32,u32)
{
    let tid=UNIT_POS as usize;let lane=tid%32;let warp=tid/32;
    #[unroll] for w in 0usize..8 {histogram[w*256+tid]=0u32;}
    sync_ruda();
    let code=ordered_key::<E,u64>(value,dtype,bits,descending,true);
    let digit=u32::cast_from((code>>shift)&255u64);
    let mut peers=plane_ballot(valid)[0usize];
    #[unroll] for bit in 0u32..8 {
        let set=(digit & comptime!(1u32<<bit))!=0u32;
        let ballot=plane_ballot(set)[0usize];
        if set {peers&=ballot;}else {peers&=ballot^u32::MAX;}
    }
    let mut rank=(peers & ((1u32<<(lane as u32))-1u32)).count_ones();
    if valid && rank==0u32 {histogram[warp*256+digit as usize]=peers.count_ones();}
    sync_ruda();
    #[unroll] for w in 0usize..8 {if w<warp {rank+=histogram[w*256+digit as usize];}}
    (digit,rank)
}

#[ruda(launch)]
fn radix_histogram<E:Numeric+RudaElement>(input:&Tensor<E>,counts:&mut Tensor<u32>,length:usize,tiles:usize,
    #[comptime] shift:u32,#[comptime] bits:u32,#[comptime] dtype:u32,#[comptime] descending:bool)
{
    let task=RUDA_POS_X as usize;let row=task/tiles;let tile=task%tiles;let tid=UNIT_POS as usize;
    let column=tile*256+tid;let valid=column<length;let mut value=E::cast_from(0);
    if valid {value=input[row*length+column];}
    let mut histogram=SharedMemory::<u32>::new(8usize*256usize);
    let _rank=radix_rank::<E>(value,valid,&mut histogram,shift,bits,dtype,descending);
    let mut count=0u32;
    #[unroll] for w in 0usize..8 {count+=histogram[w*256+tid];}
    counts[(row*tiles+tile)*256+tid]=count;
}

#[ruda(launch)]
fn radix_bases(prefix:&Tensor<u32>,bases:&mut Tensor<u32>,tiles:usize) {
    let row=RUDA_POS_X as usize;let tid=UNIT_POS as usize;
    let count=prefix[(row*tiles+tiles-1)*256+tid];
    let mut warps=SharedMemory::<u32>::new(8usize);
    let inclusive=crate::unique_ops::block_prefix(count,&mut warps);
    bases[row*256+tid]=inclusive-count;
}

#[ruda(launch)]
fn radix_scatter<E:Numeric+RudaElement>(input:&Tensor<E>,keys:&Tensor<u32>,prefix:&Tensor<u32>,bases:&Tensor<u32>,
    output:&mut Tensor<E>,output_keys:&mut Tensor<u32>,length:usize,tiles:usize,
    #[comptime] shift:u32,#[comptime] bits:u32,#[comptime] dtype:u32,#[comptime] descending:bool,#[comptime] first:bool)
{
    let task=RUDA_POS_X as usize;let row=task/tiles;let tile=task%tiles;
    let column=tile*256+UNIT_POS as usize;let valid=column<length;let mut value=E::cast_from(0);
    if valid {value=input[row*length+column];}
    let mut histogram=SharedMemory::<u32>::new(8usize*256usize);
    let ranked=radix_rank::<E>(value,valid,&mut histogram,shift,bits,dtype,descending);
    if valid {
        let digit=ranked.0 as usize;let mut destination=bases[row*256+digit]+ranked.1;
        if tile>0 {destination+=prefix[(row*tiles+tile-1)*256+digit];}
        let mut key=column as u32;
        if comptime!(!first) {key=keys[row*length+column];}
        output[row*length+destination as usize]=value;output_keys[row*length+destination as usize]=key;
    }
}

fn prepare_radix<E:Numeric+RudaElement>(c:&ComputeClient<CudaRuntime>,vs:&[View],ints:&[i64],reuse:&[View])
    ->Option<(Vec<PreparedKernel<CudaRuntime>>,Vec<View>)>
{
    let input=&vs[0];let axis=ints[1] as usize;let length=input.shape[axis];
    let inner=input.shape[axis+1..].iter().product::<usize>();let rows=input.len/length;
    let h=&c.properties().hardware;
    if ints[0]==0 || input.dtype==3 || inner!=1 || length<8192 || input.len>u32::MAX as usize
        || h.plane_size_min!=32 || h.plane_size_max!=32 || h.max_units_per_ruda<256 || h.max_ruda_dim.0<256
        || h.max_shared_memory_size<8192 {return None;}
    let tiles=length.div_ceil(256);let tasks=rows.checked_mul(tiles)?;
    if tasks>h.max_ruda_count.0 as usize || rows>h.max_ruda_count.0 as usize || tasks.checked_mul(256)? > u32::MAX as usize {return None;}
    let bits=(element_bytes(input.dtype)*8) as u32;
    let values=[reuse_or_allocate(c,&[rows,length],input.dtype,reuse,0),reuse_or_allocate(c,&[rows,length],input.dtype,reuse,1)];
    let keys=[reuse_or_allocate(c,&[rows,length],5,reuse,2),reuse_or_allocate(c,&[rows,length],5,reuse,3)];
    let counts=reuse_or_allocate(c,&[rows,tiles,256],5,reuse,4);let prefix=reuse_or_allocate(c,&[rows,tiles,256],5,reuse,5);
    let bases=reuse_or_allocate(c,&[rows,256],5,reuse,6);let mut nodes=Vec::new();let mut scan_workspace=Vec::new();
    let grid=RudaCount::Static(tasks as u32,1,1);let dim=RudaDim::new_1d(256);let mut current=1usize;
    for shift in (0..bits).step_by(8) {
        let source=if shift==0 {input} else {&values[current]};let next=1-current;
        nodes.push(radix_histogram::prepare::<E,CudaRuntime>(c,grid.clone(),dim,source.arg(),counts.arg(),length,tiles,shift,bits,input.dtype,ints[3]!=0));
        let scan_reuse=if shift==0 {reuse.get(7..).unwrap_or(&[])}else {&scan_workspace};
        let (scans,workspace)=sequence_scan::prepare_reusing(c,&counts,&prefix,14,1,scan_reuse);nodes.extend(scans);scan_workspace=workspace;
        nodes.push(radix_bases::prepare::<CudaRuntime>(c,RudaCount::Static(rows as u32,1,1),dim,prefix.arg(),bases.arg(),tiles));
        nodes.push(radix_scatter::prepare::<E,CudaRuntime>(c,grid.clone(),dim,source.arg(),keys[current].arg(),prefix.arg(),bases.arg(),
            values[next].arg(),keys[next].arg(),length,tiles,shift,bits,input.dtype,ints[3]!=0,shift==0));
        current=next;
    }
    let indices=vs.last().unwrap();let out=if ints[0]==2 {&values[current]}else {&vs[1]};
    nodes.push(emit::prepare::<E,CudaRuntime>(c,RudaCount::Static(u32::try_from(indices.len.div_ceil(128)).ok()?,1,1),RudaDim::new_1d(128),
        values[current].arg(),keys[current].arg(),out.arg(),indices.arg(),length,length,inner,indices.shape.len(),ints[0]==2));
    let mut scratch=Vec::new();scratch.extend(values);scratch.extend(keys);scratch.extend([counts,prefix,bases]);
    scratch.extend(scan_workspace);Some((nodes,scratch))
}

fn plan<E:Numeric+RudaElement>(c:&ComputeClient<CudaRuntime>,vs:&[View],ints:&[i64],reuse:&[View])->(Vec<PreparedKernel<CudaRuntime>>,Vec<View>) {
    if let Some(prepared)=prepare_radix::<E>(c,vs,ints,reuse) {return prepared;}
    let input=&vs[0];let axis=ints[1] as usize;let length=input.shape[axis];
    let inner=input.shape[axis+1..].iter().product::<usize>();let rows=input.len/length;
    if ints[0]==0 && ints[2]<=16 && length<=2048
        && (length>256 || ints[4]==0 || ints[2]<=2 && length>=16 || ints[2]<=8 && length>=64)
        && c.properties().hardware.plane_size_min==32 && c.properties().hardware.plane_size_max==32 {
        let indices=vs.last().unwrap();
        let kernel=warp_topk::prepare::<E,CudaRuntime>(c,
            RudaCount::Static(u32::try_from(rows.div_ceil(4)).expect("topk grid overflow"),1,1),RudaDim::new_1d(128),
            input.arg(),vs[1].arg(),indices.arg(),rows,inner,length,length.div_ceil(32),ints[2] as usize,indices.shape.len(),ints[3]!=0,ints[4]!=0,
            input.dtype,(element_bytes(input.dtype)*8) as u32);
        return (vec![kernel],Vec::new());
    }
    let h=&c.properties().hardware;
    if ints[0]==0 && (ints[2]<=256 || ints[4]==0) && length<=8192 && (length>256 || ints[4]==0)
        && h.plane_size_min==32 && h.plane_size_max==32 && h.max_units_per_ruda>=256 && h.max_ruda_dim.0>=256
        && h.max_shared_memory_size>=4096 && rows<=h.max_ruda_count.0 as usize {
        let indices=vs.last().unwrap();let k=ints[2] as usize;let bits=(element_bytes(input.dtype)*8) as u32;
        let padded=if ints[4]!=0 {k.next_power_of_two()}else {1};
        macro_rules! run {($u:ty)=>{block_topk::prepare::<E,$u,CudaRuntime>(c,RudaCount::Static(rows as u32,1,1),RudaDim::new_1d(256),
            input.arg(),vs[1].arg(),indices.arg(),inner,length,length.div_ceil(256),k,padded,indices.shape.len(),
            input.dtype,bits,ints[3]!=0,ints[4]!=0)};}
        let kernel=if bits==64 {run!(u64)}else {run!(u32)};
        return (vec![kernel],Vec::new());
    }
    if length<=256 && c.properties().hardware.plane_size_min==32 && c.properties().hardware.plane_size_max==32 {
        let indices=vs.last().unwrap();let out=if ints[0]==2 {input} else {&vs[1]};
        let padded=length.next_power_of_two().max(32);
        let kernel=warp_sort::prepare::<E,i64,CudaRuntime>(c,
            RudaCount::Static(u32::try_from(rows.div_ceil(4)).expect("sort grid overflow"),1,1),RudaDim::new_1d(128),
            input.arg(),out.arg(),indices.arg(),rows,inner,length,indices.shape[axis],padded/32,
            padded.trailing_zeros() as usize,indices.shape.len(),ints[3]!=0,ints[0]==2,input.dtype==3,false);
        return (vec![kernel],Vec::new());
    }
    let shape=[rows,length];let mut scratch=vec![reuse_or_allocate(c,&shape,input.dtype,reuse,0),reuse_or_allocate(c,&shape,5,reuse,1)];
    let grid=RudaCount::Static(u32::try_from(input.len.div_ceil(128)).unwrap(),1,1);let dim=RudaDim::new_1d(128);
    let warp_runs=c.properties().hardware.plane_size_min==32 && c.properties().hardware.plane_size_max==32;
    let mut nodes=if warp_runs {
        let tasks=rows.checked_mul(length.div_ceil(256)).expect("sort run count overflow");
        vec![warp_sort::prepare::<E,u32,CudaRuntime>(c,
            RudaCount::Static(u32::try_from(tasks.div_ceil(4)).expect("sort run grid overflow"),1,1),dim,
            input.arg(),scratch[0].arg(),scratch[1].arg(),tasks,inner,length,256,8,8,2,ints[3]!=0,false,input.dtype==3,true)]
    } else {vec![initialize::prepare::<E,CudaRuntime>(c,grid.clone(),dim,input.arg(),scratch[0].arg(),scratch[1].arg(),length,inner,input.dtype==3)]};
    if length>1 {scratch.push(reuse_or_allocate(c,&shape,input.dtype,reuse,2));scratch.push(reuse_or_allocate(c,&shape,5,reuse,3));}
    let mut current=0;let mut width=if warp_runs {256}else {1};
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
    prepare_reusing(c,vs,ints,&[])
}

pub(super) fn prepare_reusing(c:&ComputeClient<CudaRuntime>,vs:&[View],ints:&[i64],reuse:&[View])->(Vec<PreparedKernel<CudaRuntime>>,Vec<View>) {
    if vs.last().unwrap().len==0 {return (Vec::new(),Vec::new());}
    macro_rules! run {($e:ty)=>{plan::<$e>(c,vs,ints,reuse)};}
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
