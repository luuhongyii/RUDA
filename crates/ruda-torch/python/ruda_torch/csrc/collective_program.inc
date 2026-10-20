struct UcxProgramTransfer {
  uint32_t peer,receive;uint64_t tag,pointer;size_t bytes;
};

class CollectiveProgram {
  using IpcCollective=int (*)(void*,uint32_t,uint64_t,size_t,uint32_t,uint64_t,uint32_t);
  using IpcAllToAll=int (*)(void*,uint64_t,uint64_t,uint32_t,const size_t*,const size_t*);
  using UcxExchange=int (*)(void*,const UcxProgramTransfer*,size_t,uint64_t);
  using UcxReduce=int (*)(void*,uint64_t,uint64_t,size_t,uint32_t,uint32_t);
  c10::intrusive_ptr<c10d::ProcessGroup> pg_;
  int kind_=0,rank_=0,world_=1,device_=0;
  void* handle_=nullptr;Error error_=nullptr;
  IpcCollective ipc_=nullptr;IpcAllToAll ipc_all_=nullptr;
  UcxExchange exchange_=nullptr;UcxReduce reduce_=nullptr;
  uint64_t timeout_=0;
  std::mutex mutex_;

  void status(int code)const {TORCH_CHECK(!code,"native collective: ",error_?error_():"unknown communication error");}
  void storage(const at::Tensor& t)const {
    validate(t.device());
    TORCH_CHECK(t.layout()==c10::Layout::Strided && !t.is_conj() && !t.is_neg()
      && t.storage().data_ptr().get_deleter()==release,"native collective requires resolved RUDA storage");
  }
  static uint64_t address(const at::Tensor& t){return reinterpret_cast<uintptr_t>(t.data_ptr());}
  static uint32_t code(c10::ScalarType type) {
    switch(type) {
      case at::kFloat:return 0;case at::kDouble:return 1;case at::kInt:return 2;case at::kLong:return 3;
      case at::kByte:case at::kChar:case at::kBool:case at::kFloat8_e4m3fn:case at::kFloat8_e5m2:return 4;
      case at::kShort:case at::kHalf:case at::kBFloat16:return 5;
      default:TORCH_CHECK(false,"unsupported native collective dtype");
    }
  }
  at::Tensor alias(const at::Tensor& t)const {
    return at::from_blob(t.data_ptr(),t.sizes(),t.strides(),[owner=t](void*){},
      t.options().device(c10::Device(c10::DeviceType::CUDA,device_)).requires_grad(false));
  }
  void finish(const c10::intrusive_ptr<c10d::Work>& work)const {
    TORCH_CHECK(work && work->wait(),"native process-group work did not complete");
    c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA)->synchronizeDevice(device_);
  }
  uint64_t tag(uint64_t sequence,int step,int sender)const {
    TORCH_CHECK(sequence<(uint64_t(1)<<31) && step>=0 && step<65536,"UCX native collective tag overflow");
    return (sequence<<32)|(uint64_t(step)<<16)|uint64_t(sender);
  }
  void exchange(const std::vector<UcxProgramTransfer>& requests)const {
    stream_command(11);
    status(exchange_(handle_,requests.data(),requests.size(),timeout_));
  }
  void ring(const at::Tensor& sent,const at::Tensor& received,uint64_t sequence,int step)const {
    int previous=(rank_+world_-1)%world_,next=(rank_+1)%world_;
    auto marker=(!sent.numel() || !received.numel())?at::empty({1},sent.options().dtype(at::kByte)):sent;
    exchange({{uint32_t(previous),1,tag(sequence,step,previous),received.numel()?address(received):address(marker),size_t(received.nbytes())},
              {uint32_t(next),0,tag(sequence,step,rank_),sent.numel()?address(sent):address(marker),size_t(sent.nbytes())}});
  }
  void ucx_scatter(std::vector<at::Tensor>& chunks,uint32_t operation,uint64_t sequence)const {
    int64_t maximum=0;for(const auto& chunk:chunks)maximum=std::max(maximum,chunk.numel());
    auto scratch=at::empty({std::max(int64_t(1),maximum)},chunks[0].options());
    for(int step=0;step<world_-1;++step) {
      int sent=(rank_+world_-1-step)%world_,received=(rank_+2*world_-2-step)%world_;
      auto target=scratch.narrow(0,0,chunks[received].numel());
      ring(chunks[sent],target,sequence,step);
      status(reduce_(handle_,address(chunks[received]),address(target),size_t(target.numel()),code(target.scalar_type()),operation));
    }
  }
  void ucx_gather(std::vector<at::Tensor>& chunks,uint64_t sequence)const {
    for(int step=0;step<world_-1;++step)
      ring(chunks[(rank_+world_-step)%world_],chunks[(rank_+world_-step-1)%world_],sequence,step);
  }
  at::Tensor collective(const at::Tensor& input,uint32_t op,int64_t axis,int root,uint64_t& sequence) {
    TORCH_CHECK(op<=5 && root>=0 && root<world_,"invalid native collective policy");
    bool sharded=op>=4,reducing=op<=2 || op==5;
    if(sharded) {TORCH_CHECK(input.dim()>0,"gather/scatter needs a tensor axis");axis=at::maybe_wrap_dim(axis,input.dim());}
    auto source=sharded?input.movedim(axis,0).contiguous():input.contiguous();
    auto shape=source.sizes().vec();
    if(op==4) {TORCH_CHECK(shape[0]<=std::numeric_limits<int64_t>::max()/world_,"all-gather shape overflow");shape[0]*=world_;}
    if(op==5) {TORCH_CHECK(shape[0]%world_==0,"reduce-scatter axis must divide the group size");shape[0]/=world_;}
    auto result=at::empty(shape,source.options());
    if(world_==1) {result.copy_(source);return sharded?result.movedim(0,axis).contiguous():result;}
    if(kind_==0) {
      c10::DeviceGuard guard(c10::Device(c10::DeviceType::CUDA,device_));
      if(!sharded)result.copy_(source);
      stream_command(11);
      auto src=alias(source),dst=alias(result);
      if(op==4)finish(pg_->_allgather_base(dst,src));
      else if(op==5) {c10d::ReduceScatterOptions opts;opts.reduceOp=c10d::ReduceOp::SUM;finish(pg_->_reduce_scatter_base(dst,src,opts));}
      else {
        std::vector<at::Tensor> values{dst};
        if(op==3) {c10d::BroadcastOptions opts;opts.rootRank=root;finish(pg_->broadcast(values,opts));}
        else {c10d::AllreduceOptions opts;opts.reduceOp=op==0?c10d::ReduceOp::SUM:op==1?c10d::ReduceOp::MAX:c10d::ReduceOp::MIN;finish(pg_->allreduce(values,opts));}
      }
    } else {
      bool promote=reducing && (source.scalar_type()==at::kHalf || source.scalar_type()==at::kBFloat16);
      auto work=(promote?source.to(at::kFloat):source.clone()).reshape({-1});
      auto target=promote?at::empty(shape,source.options().dtype(at::kFloat)):result;
      auto flat=target.reshape({-1});
      auto dtype=code(work.scalar_type());TORCH_CHECK(!reducing || dtype<=3,"unsupported native ruCCL reduction dtype");
      if(kind_==1) {
        stream_command(11);
        status(ipc_(handle_,op,address(work),size_t(work.numel()),dtype,sharded?address(flat):0,uint32_t(root)));
        if(!sharded)flat.copy_(work);
      } else if(reducing) {
        std::vector<int64_t> sizes(world_,work.numel()/world_);
        for(int r=0;r<work.numel()%world_;++r)++sizes[r];
        auto chunks=work.split_with_sizes(sizes);
        ucx_scatter(chunks,op==5?0:op,sequence++);
        if(op==5)flat.copy_(chunks[rank_]);
        else {ucx_gather(chunks,sequence++);flat.copy_(work);}
      } else if(op==4) {
        auto chunks=flat.split_with_sizes(std::vector<int64_t>(world_,work.numel()));
        chunks[rank_].copy_(work);ucx_gather(chunks,sequence++);
      } else {
        flat.copy_(work);uint64_t current=sequence++;int relative=(rank_+world_-root)%world_,step=0;
        for(int mask=1;mask<world_;mask*=2,++step) {
          if(relative<mask && relative+mask<world_) {
            int peer=(relative+mask+root)%world_;
            exchange({{uint32_t(peer),0,tag(current,step,rank_),address(flat),size_t(flat.nbytes())}});
          } else if(relative>=mask && relative<2*mask) {
            int peer=(relative-mask+root)%world_;
            exchange({{uint32_t(peer),1,tag(current,step,peer),address(flat),size_t(flat.nbytes())}});
          }
        }
      }
      if(promote)result.copy_(target);
    }
    return sharded?result.movedim(0,axis).contiguous():result;
  }
  static int64_t total(const std::vector<int64_t>& counts) {
    int64_t sum=0;
    for(int64_t n:counts) {TORCH_CHECK(n>=0 && n<=std::numeric_limits<int64_t>::max()-sum,"invalid collective row counts");sum+=n;}
    return sum;
  }
  std::vector<int64_t> counts(const at::Tensor& value)const {
    TORCH_CHECK(value.scalar_type()==at::kLong && value.dim()==1 && value.numel()==world_,"exchange counts require int64[world]");
    auto host=value.to(at::kCPU).contiguous();auto data=host.const_data_ptr<int64_t>();
    std::vector<int64_t> result(data,data+world_);total(result);return result;
  }
  at::Tensor ring_metadata(const at::Tensor& position,const at::Tensor& document,const at::Tensor& valid,
      int64_t batch,int64_t length,int64_t start,int64_t total_tokens,bool query,
      int64_t global_heads=0,int64_t head_start=0,int64_t global_keys=0)const {
    auto options=position.options().dtype(at::kLong);
    auto ordinal=(at::arange(length,options)+start).expand({batch,length});
    auto batches=at::arange(batch,options).unsqueeze(1);
    auto positions=position.numel()?position:ordinal;
    auto documents=document.numel()?document:batches.expand({batch,length});
    for(const auto& t:{positions,documents})TORCH_CHECK(t.scalar_type()==at::kLong && t.sizes()==ordinal.sizes(),"ring token metadata shape differs");
    if(valid.numel()) {
      TORCH_CHECK(valid.scalar_type()==at::kBool && valid.sizes()==ordinal.sizes(),"ring validity shape differs");
      documents=at::where(valid,documents,at::full_like(documents,-1));
    }
    if(query && global_heads) {
      auto origin=((batches*global_heads+head_start)*total_tokens+ordinal)*global_keys;
      auto stride=at::full_like(origin,total_tokens*global_keys);
      return at::stack({documents,positions,origin,stride},-1).reshape({batch*length,4}).contiguous();
    }
    auto rng_index=query?ordinal+batches*total_tokens:ordinal;
    return at::stack({documents,positions,rng_index},-1).reshape({batch*length,3}).contiguous();
  }
  at::Tensor all_to_all(const at::Tensor& value,std::vector<int64_t> sends,std::vector<int64_t> receives,uint64_t& sequence) {
    TORCH_CHECK(value.dim()>0 && sends.size()==size_t(world_) && receives.size()==size_t(world_)
      && total(sends)==value.size(0) && sends[rank_]==receives[rank_],"native all-to-all partitions differ");
    auto source=value.contiguous();auto shape=source.sizes().vec();shape[0]=total(receives);
    auto result=at::empty(shape,source.options());
    if(world_==1){result.copy_(source);return result;}
    if(kind_==0) {
      c10::DeviceGuard guard(c10::Device(c10::DeviceType::CUDA,device_));stream_command(11);
      auto src=alias(source),dst=alias(result);finish(pg_->alltoall_base(dst,src,receives,sends));return result;
    }
    size_t width=1;
    for(int64_t d=1;d<source.dim();++d) {TORCH_CHECK(source.size(d)==0 || width<=std::numeric_limits<size_t>::max()/size_t(source.size(d)),"all-to-all row width overflow");width*=size_t(source.size(d));}
    std::vector<size_t> send_elements(world_),receive_elements(world_);
    for(int peer=0;peer<world_;++peer) {
      TORCH_CHECK(!width || (uint64_t(sends[peer])<=std::numeric_limits<size_t>::max()/width && uint64_t(receives[peer])<=std::numeric_limits<size_t>::max()/width),"all-to-all element count overflow");
      send_elements[peer]=size_t(sends[peer])*width;receive_elements[peer]=size_t(receives[peer])*width;
    }
    if(kind_==1) {stream_command(11);status(ipc_all_(handle_,address(source),address(result),code(source.scalar_type()),send_elements.data(),receive_elements.data()));}
    else {
      uint64_t current=sequence++;size_t src=0,dst=0;std::vector<UcxProgramTransfer> requests;
      auto marker=at::empty({1},source.options().dtype(at::kByte));
      for(int peer=0;peer<world_;++peer) {
        auto sn=send_elements[peer]*source.element_size(),rn=receive_elements[peer]*source.element_size();
        if(peer==rank_) {
          result.reshape({-1}).narrow(0,int64_t(dst/source.element_size()),int64_t(receive_elements[peer])).copy_(
            source.reshape({-1}).narrow(0,int64_t(src/source.element_size()),int64_t(send_elements[peer])));
        } else {
          requests.push_back({uint32_t(peer),1,tag(current,0,peer),rn?address(result)+dst:address(marker),rn});
          requests.push_back({uint32_t(peer),0,tag(current,0,rank_),sn?address(source)+src:address(marker),sn});
        }
        src+=sn;dst+=rn;
      }
      exchange(requests);
    }
    return result;
  }
 public:
  CollectiveProgram(c10::intrusive_ptr<c10d::ProcessGroup> pg,int device):pg_(std::move(pg)),device_(device) {
    TORCH_CHECK(pg_,"native collective requires an initialized process group");rank_=pg_->getRank();world_=pg_->getSize();
  }
  CollectiveProgram(int kind,int rank,int world,int device,const std::vector<uintptr_t>& addresses,uint64_t timeout)
    :kind_(kind),rank_(rank),world_(world),device_(device),timeout_(timeout) {
    TORCH_CHECK((kind==1 || kind==2) && world>0 && world<=65535 && rank>=0 && rank<world && addresses.size()==4,
      "invalid native ruCCL program communicator");
    for(auto address:addresses)TORCH_CHECK(address,"native ruCCL program needs live function/communicator addresses");
    handle_=reinterpret_cast<void*>(addresses[0]);error_=reinterpret_cast<Error>(addresses[1]);
    if(kind==1){ipc_=reinterpret_cast<IpcCollective>(addresses[2]);ipc_all_=reinterpret_cast<IpcAllToAll>(addresses[3]);}
    else {exchange_=reinterpret_cast<UcxExchange>(addresses[2]);reduce_=reinterpret_cast<UcxReduce>(addresses[3]);}
  }
  std::vector<at::Tensor> ring_run(bool backward,const std::vector<at::Tensor>& tensors,
      const std::vector<int64_t>& query_lengths,const std::vector<int64_t>& key_lengths,
      const std::vector<int64_t>& policy,const std::vector<double>& scalars,
      const std::vector<int64_t>& words,uint64_t sequence) {
    std::lock_guard<std::mutex> lock(mutex_);at::NoGradGuard no_grad;
    TORCH_CHECK(tensors.size()==(backward?8:9) && key_lengths.size()==size_t(world_)
      && (backward || query_lengths.size()==size_t(world_)) && (policy.size()==3 || (!backward && policy.size()==5))
      && scalars.size()==3 && words.size()==4,
      "invalid native ring program signature");
    for(const auto& t:tensors)storage(t);
    for(auto word:words)TORCH_CHECK(word>=0 && word<=0xffffffffll,"invalid ring RNG word");
    TORCH_CHECK(policy[0]==0 || policy[0]==1,"invalid ring causal flag");
    const auto &q=tensors[0],&k=tensors[1],&v=tensors[2];
    TORCH_CHECK(q.dim()==4 && k.dim()==4 && v.dim()==4,"ring program requires [B,T,H,D]");
    int64_t batch=q.size(0),queries=q.size(1),heads=q.size(2),dimension=q.size(3);
    int64_t keys=k.size(1),kv_heads=k.size(2),width=v.size(3),global_keys=total(key_lengths);
    TORCH_CHECK(batch>0 && heads>0 && kv_heads>0 && heads%kv_heads==0 && dimension>0 && width>0
      && keys==key_lengths[rank_] && k.size(0)==batch && v.size(0)==batch && v.size(1)==keys
      && v.size(2)==kv_heads && k.size(3)==dimension && q.scalar_type()==k.scalar_type()
      && q.scalar_type()==v.scalar_type() && global_keys<=0xffffffffll,"ring program head/partition geometry differs");
    uint64_t seed=uint64_t(words[0])|(uint64_t(words[1])<<32),counter=uint64_t(words[2])|(uint64_t(words[3])<<32);
    auto call=[&](uint32_t op,const std::vector<at::Tensor>& args) {
      ring_attention_call(op,args,batch,uint32_t(global_keys),bool(policy[0]),policy[1],policy[2],
        scalars[0],scalars[1],scalars[2],seed,counter);
    };
    auto query=q.contiguous().reshape({batch*queries,heads,dimension});
    auto key=k.contiguous().reshape({batch*keys,kv_heads,dimension});
    auto value=v.contiguous().reshape({batch*keys,kv_heads,width});
    auto rotate=[&](const std::vector<at::Tensor>& block,int owner) {
      std::vector<int64_t> sends(world_,0),receives(world_,0);
      sends[(rank_+1)%world_]=block[0].size(0);receives[(rank_+world_-1)%world_]=batch*key_lengths[owner];
      std::vector<at::Tensor> result;
      for(const auto& tensor:block)result.push_back(all_to_all(tensor,sends,receives,sequence));
      return result;
    };
    if(backward) {
      auto gradient=tensors[3].contiguous().reshape({batch*queries,heads,width});
      auto dq=at::zeros(query.sizes(),query.options().dtype(at::kFloat));
      auto dk=at::zeros(key.sizes(),key.options().dtype(at::kFloat));
      auto dv=at::zeros(value.sizes(),value.options().dtype(at::kFloat));
      std::vector<at::Tensor> block{key,value,tensors[7].contiguous(),dk,dv};
      for(int step=0;step<world_;++step) {
        call(1,{query,block[0],block[1],tensors[6],block[2],gradient,tensors[4],tensors[5],dq,block[3],block[4]});
        if(world_>1)block=rotate(block,(rank_+2*world_-step-1)%world_);
      }
      return {dq.to(q.scalar_type()).reshape(q.sizes()),block[3].to(k.scalar_type()).reshape(k.sizes()),
              block[4].to(v.scalar_type()).reshape(v.sizes())};
    }
    int64_t global_queries=total(query_lengths),query_start=0,key_start=0;
    const bool indexed=policy.size()==5;
    const int64_t global_heads=indexed?policy[3]:0,head_start=indexed?policy[4]:0;
    TORCH_CHECK(queries==query_lengths[rank_],"ring query partition differs");
    if(indexed) {
      TORCH_CHECK(global_heads>0 && head_start>=0 && head_start<=global_heads-heads,"ring global head interval differs");
      int64_t scores=1;
      for(auto size:{batch,global_heads,global_queries,global_keys}) {
        TORCH_CHECK(!size || scores<=std::numeric_limits<int64_t>::max()/size,"ring global RNG index overflow");
        scores*=size;
      }
      if(scalars[1]>0 && scalars[1]<1) {
        uint64_t blocks=uint64_t(scores)/4+uint64_t(scores%4!=0);
        TORCH_CHECK(counter<=std::numeric_limits<uint64_t>::max()-blocks,"ring RNG counter exhausted");
      }
    } else TORCH_CHECK(global_queries<=0xffffffffll/batch/heads,"ring global query indexing overflow");
    for(int peer=0;peer<rank_;++peer){query_start+=query_lengths[peer];key_start+=key_lengths[peer];}
    auto qmeta=ring_metadata(tensors[3],tensors[5],tensors[7],batch,queries,query_start,global_queries,true,
      global_heads,head_start,global_keys);
    auto kmeta=ring_metadata(tensors[4],tensors[6],tensors[8],batch,keys,key_start,global_keys,false);
    auto accumulator=at::zeros({batch*queries,heads,width},q.options().dtype(at::kFloat));
    auto stats=at::zeros({batch*queries*heads,2},q.options().dtype(at::kFloat));
    std::vector<at::Tensor> block{key,value,kmeta};
    for(int step=0;step<world_;++step) {
      call(0,{query,block[0],block[1],qmeta,block[2],accumulator,stats});
      if(step+1<world_)block=rotate(block,(rank_+2*world_-step-1)%world_);
    }
    auto output=at::empty({batch*queries,heads,width},q.options());call(2,{accumulator,stats,output});
    auto state=at::empty({int64_t(5+world_)},at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    auto data=state.mutable_data_ptr<int64_t>();data[0]=global_keys;
    std::copy(words.begin(),words.end(),data+1);std::copy(key_lengths.begin(),key_lengths.end(),data+5);
    return {output.reshape({batch,queries,heads,width}),accumulator,stats,qmeta,kmeta,state};
  }
  std::vector<at::Tensor> run(int op,const std::vector<at::Tensor>& tensors,const std::vector<int64_t>& integers,uint64_t sequence) {
    std::lock_guard<std::mutex> lock(mutex_);at::NoGradGuard no_grad;
    TORCH_CHECK(!tensors.empty() && integers.size()==1,"invalid native collective program signature");
    for(const auto& t:tensors)storage(t);
    const auto& x=tensors[0];int64_t option=integers[0];
    if(op==16 || op==17 || op==18) {
      TORCH_CHECK(tensors.size()==1 && option>0 && option<=world_ && world_%option==0,"invalid TP replica geometry");
      int replicas=int(option),lane=rank_%replicas;
      if(op==17) {
        TORCH_CHECK(x.dim()>0,"unique TP gather needs a feature axis");
        auto full=collective(x,4,-1,0,sequence);
        auto shape=x.sizes().vec();shape.pop_back();
        shape.insert(shape.end(),{world_/replicas,replicas,x.size(-1)});
        auto result=full.reshape(shape).select(-2,0);
        shape=x.sizes().vec();shape.back()*=world_/replicas;
        return {result.reshape(shape).contiguous().clone()};
      }
      if(replicas==1 || !x.numel())return {x.clone()};
      TORCH_CHECK(op==18 || x.is_floating_point(),"TP replica sum requires floating values");
      auto work=(op==16 && (x.scalar_type()==at::kHalf || x.scalar_type()==at::kBFloat16)?x.to(at::kFloat):x.clone()).contiguous().reshape({1,-1});
      std::vector<int> steps;
      for(int step=1;step<replicas;step*=2) {
        steps.push_back(step);if(op==18)continue;
        std::vector<int64_t> sends(world_,0),receives(world_,0);
        if(lane%(2*step)==step)sends[rank_-step]=1;
        else if(lane%(2*step)==0 && lane+step<replicas)receives[rank_+step]=1;
        auto received=all_to_all(work.narrow(0,0,total(sends)),sends,receives,sequence);
        if(total(receives))work=work+received;
      }
      for(auto it=steps.rbegin();it!=steps.rend();++it) {
        int step=*it;std::vector<int64_t> sends(world_,0),receives(world_,0);
        if(lane%(2*step)==0 && lane+step<replicas)sends[rank_+step]=1;
        else if(lane%(2*step)==step)receives[rank_-step]=1;
        auto received=all_to_all(work.narrow(0,0,total(sends)),sends,receives,sequence);
        if(total(receives))work=received;
      }
      return {work.reshape(x.sizes()).to(x.scalar_type())};
    }
    if(op==0)return {x.clone()};
    if(op==1 || op==11)return {collective(x,0,0,0,sequence)};
    if(op==2 || op==4)return {collective(x,4,option,0,sequence)};
    if(op==3) {
      int64_t axis=at::maybe_wrap_dim(option,x.dim());TORCH_CHECK(x.size(axis)%world_==0,"TP slice axis must divide group size");
      int64_t size=x.size(axis)/world_;return {x.narrow(axis,rank_*size,size).contiguous().clone()};
    }
    if(op==5)return {collective(x,5,option,0,sequence)};
    if(op==6) {
      std::vector<int64_t> sends(world_,0),receives(world_,0);
      int64_t from=rank_-option,to=rank_+option;
      if(to>=0 && to<world_)sends[to]=1;if(from>=0 && from<world_)receives[from]=1;
      auto source=x.contiguous().reshape({1,x.numel()});
      auto result=all_to_all(source.narrow(0,0,total(sends)),sends,receives,sequence);
      return {total(receives)?result.reshape(x.sizes()):at::zeros_like(x)};
    }
    if(op==7)return {collective(x,3,0,0,sequence)};
    if(op==8){auto result=collective(x,0,0,0,sequence);if(rank_!=0)result.zero_();return {result};}
    if(op==9){TORCH_CHECK(option>=0 && option<=2,"invalid reduction operation");return {collective(x,uint32_t(option),0,0,sequence)};}
    if(op==10){auto maximum=x.numel()?x.amax().reshape({1}):at::full({1},-1,x.options());return {collective(maximum,1,0,0,sequence)};}
    if(op==12) {
      TORCH_CHECK(tensors.size()==2,"context loss needs sum and count");
      auto sum=collective(x,0,0,0,sequence),count=collective(tensors[1],0,0,0,sequence);
      auto denominator=at::where(count.gt(0),count,at::ones_like(count)).to(sum.scalar_type());return {sum/denominator,count};
    }
    if(op==13) {
      TORCH_CHECK(tensors.size()==3 && x.dim()==2 && tensors[1].scalar_type()==at::kLong
        && tensors[1].dim()==1 && tensors[1].size(0)==x.size(0),"expert dispatch requires rows, IDs and counts");
      auto sends=counts(tensors[2]);TORCH_CHECK(total(sends)==x.size(0),"expert send counts differ from rows");
      auto matrix=collective(tensors[2],4,0,0,sequence).reshape({world_,world_});
      auto receive_tensor=matrix.select(1,rank_).contiguous();auto receives=counts(receive_tensor);
      auto rows=all_to_all(x,sends,receives,sequence),ids=all_to_all(tensors[1],sends,receives,sequence);
      return {rows,ids,receive_tensor};
    }
    TORCH_CHECK((op==14 || op==15) && tensors.size()==4,"invalid expert inverse program");
    auto sends=counts(tensors[1]),receives=counts(tensors[2]);
    TORCH_CHECK(tensors[3].dim()>0 && total(op==14?sends:receives)==tensors[3].size(0),"expert original row count differs");
    return {op==14?all_to_all(x,receives,sends,sequence):all_to_all(x,sends,receives,sequence)};
  }
};
