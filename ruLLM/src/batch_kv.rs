use crate::{CausalModelLimits,ContinuousBatchScheduler,GenerationError,KvPageId,PagedKvCacheConfig,RequestId,ScheduledBatch};
use ruda::runtime::server::ComputeServer;
use ruda_tensor::api::{backend::Backend,DType,Int,Tensor,TensorData};
use ruda_tensor::{DeviceOps,TensorPrimitive};
use ruda_tensor_device::{BoolElement,DeviceBackend,DeviceRuntime,FloatElement,IntElement};
use std::collections::{BTreeMap,BTreeSet};

#[derive(Clone)]
struct Arena<B:Backend> {key:Tensor<B,4>,value:Tensor<B,4>}

#[derive(Clone)]
pub struct DenseBatchDelta<B:Backend> {start:usize,count:usize,layers:Vec<(Tensor<B,4>,Tensor<B,4>)>}

#[derive(Clone)]
pub struct DenseBatchPrefix<B:Backend> {position:usize,layers:Vec<(Tensor<B,4>,Tensor<B,4>)>}
impl<B:Backend> DenseBatchPrefix<B> {
    pub(crate) fn position(&self)->usize {self.position}
    pub(crate) fn bytes(&self)->usize {
        self.layers.iter().flat_map(|(k,v)|[k,v]).fold(0usize,|bytes,t|bytes.saturating_add(t.dims().iter().product::<usize>().saturating_mul(t.dtype().size())))
    }
    pub(crate) fn validate(&self,geometry:&[(usize,usize,usize,DType)],limit:usize)->Result<(),GenerationError> {
        if self.position==0 || self.position>limit || self.layers.len()!=geometry.len() {return Err(GenerationError("dense prefix layer count/position differs".into()));}
        for ((key,value),&(heads,kd,vd,dtype)) in self.layers.iter().zip(geometry) {
            if key.dims()!=[1,heads,self.position,kd] || value.dims()!=[1,heads,self.position,vd]
                || key.dtype()!=dtype || value.dtype()!=dtype || key.device()!=value.device() {
                return Err(GenerationError("dense prefix projection geometry or dtype differs".into()));
            }
        }
        Ok(())
    }
    pub(crate) fn encode(&self)->Result<Vec<u8>,GenerationError> {
        let layers=self.layers.iter().map(|(k,v)|Ok((k.clone().try_into_data().map_err(|e|GenerationError(e.to_string()))?,
            v.clone().try_into_data().map_err(|e|GenerationError(e.to_string()))?))).collect::<Result<Vec<_>,GenerationError>>()?;
        rmp_serde::to_vec(&(1u32,self.position,layers)).map_err(|e|GenerationError(e.to_string()))
    }
    pub(crate) fn decode(bytes:&[u8],geometry:&[(usize,usize,usize,DType)],limit:usize,device:&B::Device)->Result<Self,GenerationError> {
        let (version,position,layers):(u32,usize,Vec<(TensorData,TensorData)>)=rmp_serde::from_slice(bytes).map_err(|e|GenerationError(e.to_string()))?;
        if version!=1 || position==0 || position>limit || layers.len()!=geometry.len() {return Err(GenerationError("invalid dense prefix record geometry".into()));}
        for ((key,value),&(heads,kd,vd,dtype)) in layers.iter().zip(geometry) {
            for (tensor,width) in [(key,kd),(value,vd)] {
                if tensor.shape.as_slice()!=[1,heads,position,width] || tensor.dtype!=dtype
                    || heads.checked_mul(position).and_then(|n|n.checked_mul(width)).and_then(|n|n.checked_mul(dtype.size()))!=Some(tensor.as_bytes().len()) {
                    return Err(GenerationError("invalid dense prefix native tensor payload".into()));
                }
            }
        }
        let layers=layers.into_iter().map(|(key,value)|{let dtype=key.dtype;
            (Tensor::from_data(key,(device,dtype)),Tensor::from_data(value,(device,dtype)))}).collect();
        Ok(Self{position,layers})
    }
}

/// Shared paged device storage for dense/GQA adapters. Forks isolate appends
/// through DevicePlan's copy-on-write contract; no per-request model forwards.
#[derive(Clone)]
pub struct DenseBatchCache<B:Backend> {
    config:PagedKvCacheConfig,
    arenas:Vec<Option<Arena<B>>>,
    positions:BTreeMap<RequestId,usize>,
    tables:BTreeMap<RequestId,Vec<KvPageId>>,
}
impl<B:Backend> DenseBatchCache<B> {
    pub(crate) fn new(config:PagedKvCacheConfig,layers:usize)->Self {
        Self{config,arenas:(0..layers).map(|_|None).collect(),positions:BTreeMap::new(),tables:BTreeMap::new()}
    }
    pub fn resident_requests(&self)->usize {self.positions.len()}
    pub fn sequence_length(&self,id:RequestId)->Option<usize> {self.positions.get(&id).copied()}
    pub(crate) fn config(&self)->PagedKvCacheConfig {self.config}
    pub(crate) fn truncate(&mut self,id:RequestId,position:usize)->Result<(),GenerationError> {
        let old=self.positions.get_mut(&id).ok_or_else(||GenerationError("unknown dense truncate request".into()))?;
        if position>*old {return Err(GenerationError("dense cache cannot truncate forward".into()));}
        *old=position;self.tables.get_mut(&id).unwrap().truncate(position.div_ceil(self.config.block_size));Ok(())
    }
    pub(crate) fn capture_delta(&self,id:RequestId,start:usize,end:usize)->Result<DenseBatchDelta<B>,GenerationError> {
        if start>=end || self.sequence_length(id).is_none_or(|position|end>position) {return Err(GenerationError("invalid dense delta span".into()));}
        let block=self.config.block_size;let first=start/block;let last=end.div_ceil(block);let table=&self.tables[&id][first..last];
        let pages=table.iter().map(|p|i32::try_from(p.0).map_err(|e|GenerationError(e.to_string()))).collect::<Result<Vec<_>,_>>()?;
        let mut layers=Vec::with_capacity(self.arenas.len());
        for arena in &self.arenas {
            let arena=arena.as_ref().ok_or_else(||GenerationError("incomplete dense delta layers".into()))?;
            let extract=|tensor:Tensor<B,4>| {
                let [_,block,h,d]=tensor.dims();let offset=start%block;
                let ids=Tensor::<B,1,Int>::from_data(TensorData::new(pages.clone(),[pages.len()]),&tensor.device());
                tensor.select(0,ids).reshape([pages.len()*block,h,d]).slice([offset..offset+end-start,0..h,0..d])
                    .swap_dims(0,1).unsqueeze_dim::<4>(0)
            };
            layers.push((extract(arena.key.clone()),extract(arena.value.clone())));
        }
        Ok(DenseBatchDelta{start,count:end-start,layers})
    }
    pub(crate) fn append_delta(&mut self,id:RequestId,table:&[KvPageId],delta:&DenseBatchDelta<B>,offset:usize)
        ->Result<(),GenerationError> {
        let position=delta.start.checked_add(offset).ok_or_else(||GenerationError("dense delta position overflow".into()))?;
        let block=self.config.block_size;
        if offset>=delta.count || self.sequence_length(id)!=Some(position) || position>=self.config.max_sequence_length
            || table.len()!=(position+1).div_ceil(block) || delta.layers.len()!=self.arenas.len()
            || table.iter().any(|p|p.0 as usize>=self.config.num_pages) || table.iter().collect::<BTreeSet<_>>().len()!=table.len()
            || !table.starts_with(&self.tables[&id]) || self.tables.iter().any(|(&other,pages)|other!=id && pages.iter().any(|p|table.contains(p))) {
            return Err(GenerationError("invalid dense lookahead delta destination".into()));
        }
        let page=table[position/block].0 as usize;let slot=position%block;
        for (index,(key,value)) in delta.layers.iter().enumerate() {
            let arena=self.arenas[index].as_ref().ok_or_else(||GenerationError("missing dense delta destination arena".into()))?;
            let [_,h,n,kd]=key.dims();let vd=value.dims()[3];
            if n!=delta.count || value.dims()!=[1,h,n,vd] || arena.key.dims()!=[self.config.num_pages,block,h,kd]
                || arena.value.dims()!=[self.config.num_pages,block,h,vd] || key.dtype()!=arena.key.dtype() || value.dtype()!=arena.value.dtype() {
                return Err(GenerationError("dense lookahead delta geometry differs".into()));
            }
        }
        for (index,(key,value)) in delta.layers.iter().enumerate() {
            let arena=self.arenas[index].take().unwrap();let h=key.dims()[1];let kd=key.dims()[3];let vd=value.dims()[3];
            let key=arena.key.slice_assign([page..page+1,slot..slot+1,0..h,0..kd],key.clone().slice([0..1,0..h,offset..offset+1,0..kd]).swap_dims(1,2));
            let value=arena.value.slice_assign([page..page+1,slot..slot+1,0..h,0..vd],value.clone().slice([0..1,0..h,offset..offset+1,0..vd]).swap_dims(1,2));
            self.arenas[index]=Some(Arena{key,value});
        }
        self.positions.insert(id,position+1);self.tables.insert(id,table.to_vec());Ok(())
    }
    pub(crate) fn validate(&self,batch:&ScheduledBatch,limits:CausalModelLimits,layers:usize)->Result<(),GenerationError> {
        let error=|message:&str|GenerationError(message.into());
        self.config.validate().map_err(|e|error(&e.to_string()))?;
        if batch.batch_size()==0 || batch.next_n()==0 || self.arenas.len()!=layers {
            return Err(error("invalid dense batch/cache geometry"));
        }
        let mut requests=BTreeSet::new();let mut owners=BTreeMap::new();
        for (id,table) in &self.tables {for page in table {owners.insert(*page,*id);}}
        for row in &batch.sequences {
            if !requests.insert(row.request_id) || row.token_ids.len()!=batch.next_n()
                || row.start_position.checked_add(batch.next_n())!=Some(row.context_length)
                || row.context_length>self.config.max_sequence_length || row.context_length>limits.max_sequence_length
                || self.positions.get(&row.request_id).copied().unwrap_or(0)!=row.start_position
                || row.block_table.len()!=row.context_length.div_ceil(self.config.block_size)
                || row.token_ids.iter().any(|&token|token<0 || token as usize>=limits.vocab_size)
                || self.tables.get(&row.request_id).is_some_and(|old|!row.block_table.starts_with(old)) {
                return Err(error("dense batch request position, tokens or page table differ"));
            }
            let mut unique=BTreeSet::new();
            for &page in &row.block_table {
                if page.0 as usize>=self.config.num_pages || !unique.insert(page)
                    || owners.insert(page,row.request_id).is_some_and(|owner|owner!=row.request_id) {
                    return Err(error("dense batch pages overlap or exceed the arena"));
                }
            }
        }
        Ok(())
    }
    pub(crate) fn commit(&mut self,batch:&ScheduledBatch) {
        for row in &batch.sequences {
            self.positions.insert(row.request_id,row.context_length);
            self.tables.insert(row.request_id,row.block_table.clone());
        }
    }
    pub(crate) fn retain(&mut self,scheduler:&ContinuousBatchScheduler) {
        self.positions.retain(|id,_|scheduler.kv_cache().contains_sequence(*id));
        self.tables.retain(|id,_|self.positions.contains_key(id));
    }
    pub(crate) fn capture_prefix(&self,id:RequestId)->Result<DenseBatchPrefix<B>,GenerationError> {
        let position=*self.positions.get(&id).ok_or_else(||GenerationError("unknown dense prefix request".into()))?;
        let table=self.tables.get(&id).ok_or_else(||GenerationError("missing dense prefix page table".into()))?;
        let pages=table.iter().map(|page|i32::try_from(page.0).map_err(|e|GenerationError(e.to_string()))).collect::<Result<Vec<_>,_>>()?;
        let mut layers=Vec::with_capacity(self.arenas.len());
        for arena in &self.arenas {
            let arena=arena.as_ref().ok_or_else(||GenerationError("incomplete dense prefix layer".into()))?;
            let extract=|tensor:Tensor<B,4>| {
                let [_,block,heads,width]=tensor.dims();
                let indices=Tensor::<B,1,Int>::from_data(TensorData::new(pages.clone(),[pages.len()]),&tensor.device());
                tensor.select(0,indices).reshape([pages.len()*block,heads,width])
                    .slice([0..position,0..heads,0..width]).swap_dims(0,1).unsqueeze_dim::<4>(0)
            };
            layers.push((extract(arena.key.clone()),extract(arena.value.clone())));
        }
        Ok(DenseBatchPrefix{position,layers})
    }
    pub(crate) fn restore_prefix(&mut self,id:RequestId,table:&[KvPageId],prefix:&DenseBatchPrefix<B>)->Result<(),GenerationError> {
        if self.positions.contains_key(&id) || prefix.layers.len()!=self.arenas.len() || prefix.position>self.config.max_sequence_length
            || table.len()!=prefix.position.div_ceil(self.config.block_size) || table.iter().any(|p|p.0 as usize>=self.config.num_pages)
            || table.iter().collect::<BTreeSet<_>>().len()!=table.len()
            || self.tables.values().any(|old|old.iter().any(|p|table.contains(p))) {
            return Err(GenerationError("invalid dense prefix destination pages".into()));
        }
        for (index,(key,value)) in prefix.layers.iter().enumerate() {
            let [_,heads,_,kd]=key.dims();let vd=value.dims()[3];let block=self.config.block_size;
            let old=self.arenas[index].take().unwrap_or_else(||Arena {
                key:Tensor::empty([self.config.num_pages,block,heads,kd],(&key.device(),key.dtype())),
                value:Tensor::empty([self.config.num_pages,block,heads,vd],(&value.device(),value.dtype())),
            });
            let mut target_key=old.key;let mut target_value=old.value;
            for (page,&id) in table.iter().enumerate() {
                let start=page*block;let count=block.min(prefix.position-start);let physical=id.0 as usize;
                target_key=target_key.slice_assign([physical..physical+1,0..count,0..heads,0..kd],
                    key.clone().slice([0..1,0..heads,start..start+count,0..kd]).swap_dims(1,2));
                target_value=target_value.slice_assign([physical..physical+1,0..count,0..heads,0..vd],
                    value.clone().slice([0..1,0..heads,start..start+count,0..vd]).swap_dims(1,2));
            }
            self.arenas[index]=Some(Arena{key:target_key,value:target_value});
        }
        self.positions.insert(id,prefix.position);self.tables.insert(id,table.to_vec());Ok(())
    }
}
impl<R,F,I,BT> DenseBatchCache<DeviceBackend<R,F,I,BT>>
where R:DeviceRuntime,R::Server:ComputeServer,R::Device:DeviceOps,F:FloatElement,I:IntElement,BT:BoolElement {
    pub(crate) fn attention(&mut self,layer:usize,batch:&ScheduledBatch,
        q:Tensor<DeviceBackend<R,F,I,BT>,4>,k:Tensor<DeviceBackend<R,F,I,BT>,4>,
        v:Tensor<DeviceBackend<R,F,I,BT>,4>,scale:f32)->Result<Tensor<DeviceBackend<R,F,I,BT>,4>,GenerationError> {
        self.attention_queries(layer,batch,batch,q,k,v,scale,true)
    }
    pub(crate) fn attention_queries(&mut self,layer:usize,writes:&ScheduledBatch,queries:&ScheduledBatch,
        q:Tensor<DeviceBackend<R,F,I,BT>,4>,k:Tensor<DeviceBackend<R,F,I,BT>,4>,
        v:Tensor<DeviceBackend<R,F,I,BT>,4>,scale:f32,causal:bool)->Result<Tensor<DeviceBackend<R,F,I,BT>,4>,GenerationError> {
        use rudnn::paged_attention::{HostPlan,DevicePlan};
        let [b,h,s,d]=q.dims();let [kb,kh,ks,kd]=k.dims();let [vb,vh,vs,dv]=v.dims();
        let error=|message:&str|GenerationError(format!("dense paged batch: {message}"));
        if b!=queries.batch_size() || b!=writes.batch_size() || s!=queries.next_n() || [kb,ks,kd]!=[b,writes.next_n(),d]
            || [vb,vh,vs]!=[b,kh,ks] || h==0 || kh==0 || h%kh!=0 || layer>=self.arenas.len()
            || writes.sequences.iter().zip(&queries.sequences).any(|(w,q)|w.request_id!=q.request_id || w.block_table!=q.block_table || w.context_length!=q.context_length) {
            return Err(error("Q/K/V geometry mismatch"));
        }
        let mut tables=Vec::with_capacity(b);let mut lengths=Vec::with_capacity(b);
        let mut ids=Vec::with_capacity(b*ks);let mut positions=Vec::with_capacity(b*ks);
        let mut query_ids=Vec::with_capacity(b*s);let mut query_positions=Vec::with_capacity(b*s);
        for (index,row) in writes.sequences.iter().enumerate() {
            tables.push(row.block_table.iter().map(|p|p.0).collect());
            lengths.push(u32::try_from(row.context_length).map_err(|_|error("context overflow"))?);
            for step in 0..ks {
                ids.push(u32::try_from(index).map_err(|_|error("batch overflow"))?);
                positions.push(u32::try_from(row.start_position+step).map_err(|_|error("position overflow"))?);
            }
            for step in 0..s {query_ids.push(index as u32);query_positions.push(u32::try_from(queries.sequences[index].start_position+step).map_err(|_|error("query position overflow"))?);}
        }
        let host=HostPlan::new(self.config.block_size,self.config.num_pages,&tables,&lengths,&ids,&positions)
            .map_err(|e|error(&e.0))?;
        host.validate_writes().map_err(|e|error(&e.0))?;
        let q=q.swap_dims(1,2).reshape([b*s,h,d]).into_primitive().tensor();
        let query_host=if writes==queries {None} else {Some(HostPlan::new(self.config.block_size,self.config.num_pages,&tables,&lengths,&query_ids,&query_positions).map_err(|e|error(&e.0))?)};
        let k=k.swap_dims(1,2).reshape([b*ks,kh,d]).into_primitive().tensor();
        let v=v.swap_dims(1,2).reshape([b*ks,kh,dv]).into_primitive().tensor();
        let plan=DevicePlan::upload(host,&q);
        let arena=self.arenas[layer].take().unwrap_or_else(||Arena {
            key:Tensor::empty([self.config.num_pages,self.config.block_size,kh,d],(&q.device,q.dtype)),
            value:Tensor::empty([self.config.num_pages,self.config.block_size,kh,dv],(&q.device,q.dtype)),
        });
        let (keys,values)=plan.append(k,v,arena.key.into_primitive().tensor(),arena.value.into_primitive().tensor())
            .map_err(|e|error(&e.0))?;
        let result=if let Some(query_host)=query_host {
            DevicePlan::upload(query_host,&q).attention(q,keys.clone(),values.clone(),scale,causal)
        } else {plan.attention(q,keys.clone(),values.clone(),scale,causal)}.map_err(|e|error(&e.0))?;
        self.arenas[layer]=Some(Arena{key:Tensor::from_primitive(TensorPrimitive::Float(keys)),
            value:Tensor::from_primitive(TensorPrimitive::Float(values))});
        Ok(Tensor::<DeviceBackend<R,F,I,BT>,3>::from_primitive(TensorPrimitive::Float(result))
            .reshape([b,s,h,dv]).swap_dims(1,2))
    }
}

macro_rules! dense_prefix_model {
    ($model:ident,|$this:ident|$geometry:expr)=>{
        impl<R,F,I,BT> crate::PrefixBatchModel<ruda_tensor_device::DeviceBackend<R,F,I,BT>> for $model<ruda_tensor_device::DeviceBackend<R,F,I,BT>>
        where R:ruda_tensor_device::DeviceRuntime,R::Server:ruda::runtime::server::ComputeServer,R::Device:ruda_tensor::DeviceOps,
            F:ruda_tensor_device::FloatElement,I:ruda_tensor_device::IntElement,BT:ruda_tensor_device::BoolElement {
            type Prefix=crate::DenseBatchPrefix<ruda_tensor_device::DeviceBackend<R,F,I,BT>>;
            fn capture_prefix(&self,cache:&Self::Cache,id:crate::RequestId)->Result<Self::Prefix,crate::GenerationError> {cache.capture_prefix(id)}
            fn prefix_position(&self,prefix:&Self::Prefix)->usize {prefix.position()}
            fn prefix_bytes(&self,prefix:&Self::Prefix)->usize {prefix.bytes()}
            fn restore_prefix(&self,cache:&mut Self::Cache,id:crate::RequestId,table:&[crate::KvPageId],prefix:&Self::Prefix)->Result<(),crate::GenerationError> {
                let $this=self;let geometry=$geometry;prefix.validate(&geometry,self.batch_limits().max_sequence_length)?;
                cache.restore_prefix(id,table,prefix)
            }
            fn encode_prefix(&self,prefix:&Self::Prefix,_model_id:&str)->Result<Vec<u8>,crate::GenerationError> {prefix.encode()}
            fn decode_prefix(&self,bytes:&[u8],_model_id:&str,device:&R::Device)->Result<Self::Prefix,crate::GenerationError> {
                let $this=self;let geometry=$geometry;crate::DenseBatchPrefix::decode(bytes,&geometry,self.batch_limits().max_sequence_length,device)
            }
        }
    }
}
pub(crate) use dense_prefix_model;

macro_rules! dense_speculative_model {
    ($model:ident)=>{
        impl<R,F,I,BT> crate::SpeculativeBatchModel<ruda_tensor_device::DeviceBackend<R,F,I,BT>> for $model<ruda_tensor_device::DeviceBackend<R,F,I,BT>>
        where R:ruda_tensor_device::DeviceRuntime,R::Server:ruda::runtime::server::ComputeServer,R::Device:ruda_tensor::DeviceOps,
            F:ruda_tensor_device::FloatElement,I:ruda_tensor_device::IntElement,BT:ruda_tensor_device::BoolElement {
            type Delta=crate::DenseBatchDelta<ruda_tensor_device::DeviceBackend<R,F,I,BT>>;
            fn forward_batch_block(&self,batch:&crate::ScheduledBatch,cache:&mut Self::Cache,device:&R::Device)
                ->Result<ruda_tensor::api::Tensor<ruda_tensor_device::DeviceBackend<R,F,I,BT>,3>,crate::GenerationError> {
                self.forward_batch_logits(batch,cache,device,false)
            }
            fn capture_batch_delta(&self,cache:&Self::Cache,id:crate::RequestId,start:usize,end:usize)->Result<Self::Delta,crate::GenerationError> {
                cache.capture_delta(id,start,end)
            }
            fn truncate_batch_cache(&self,cache:&mut Self::Cache,id:crate::RequestId,position:usize)->Result<(),crate::GenerationError> {
                cache.truncate(id,position)
            }
            fn append_batch_delta(&self,cache:&mut Self::Cache,id:crate::RequestId,table:&[crate::KvPageId],delta:&Self::Delta,offset:usize)
                ->Result<(),crate::GenerationError> {cache.append_delta(id,table,delta,offset)}
        }
    }
}
pub(crate) use dense_speculative_model;
