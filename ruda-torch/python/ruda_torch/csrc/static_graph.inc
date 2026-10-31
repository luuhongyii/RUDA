// Included inside backend.cpp's anonymous namespace, after Argument.
// Optional extension API 3; deliberately does not change base tensor ABI 10.
struct StaticNode { uint32_t op, a, b; float scalar; };
static_assert(sizeof(StaticNode)==16 && alignof(StaticNode)==4, "static graph ABI layout");
using StaticGraphCommand = int (*)(uint32_t,void**,const Descriptor*,size_t,
                                   const StaticNode*,size_t,size_t,uint32_t,uint64_t*);
StaticGraphCommand static_graph_native=nullptr;

class StaticGraphBridge {
  struct Snapshot {
    void* allocation;
    void* address;
    size_t offset;
    c10::ScalarType dtype;
    std::vector<int64_t> shape,strides;
  };
  void* plan_=nullptr;
  std::vector<at::Tensor> tensors_;
  std::vector<Snapshot> snapshots_;
  size_t inputs_=0;
  uint64_t stream_=0, edges_=0;
  bool tracked_=false;
  std::mutex mutex_;
  uint64_t command(uint32_t op) {
    uint64_t value=0;
    check(static_graph_native(op,&plan_,nullptr,0,nullptr,0,0,0,&value));
    return value;
  }
  void check_bindings() const {
    TORCH_CHECK(plan_,"static graph is closed");
    TORCH_CHECK(stream_command(0)==stream_,"static graph replay must use its creation stream");
    for(size_t i=0;i<tensors_.size();++i) {
      const auto& t=tensors_[i]; const auto& s=snapshots_[i];
      TORCH_CHECK(!t.requires_grad(),"static graph is inference-only");
      // Replay needs identity/metadata checks, not fresh Descriptor vectors.
      // Keep every safety guard from Argument, then compare borrowed array views.
      validate(t.device());
      TORCH_CHECK(!t.is_conj() && !t.is_neg(), "RUDA does not support unresolved view bits");
      const auto storage=t.storage();
      TORCH_CHECK(storage.data_ptr().get_deleter()==release,"tensor is not allocated by RUDA");
      TORCH_CHECK(storage.data_ptr().get_context()==s.allocation
        && size_t(t.storage_offset())*t.element_size()==s.offset
        && t.data_ptr()==s.address && t.scalar_type()==s.dtype
        && t.sizes()==c10::IntArrayRef(s.shape) && t.strides()==c10::IntArrayRef(s.strides),
        "static graph binding changed: keep storage, shape, strides and dtype fixed");
    }
    for(size_t i=inputs_;i<tensors_.size();++i) {
      TORCH_CHECK(!tensors_[i].is_inference() || c10::InferenceMode::is_enabled(),
                  "replay writing inference tensors requires torch.inference_mode()");
    }
  }
 public:
  StaticGraphBridge(const std::vector<at::Tensor>& tensors,size_t inputs,
                    const std::vector<uint32_t>& words,const std::vector<float>& scalars,
                    bool tracked,bool infer):tensors_(tensors),inputs_(inputs),tracked_(tracked) {
    TORCH_CHECK(static_graph_native,"native static graph API is unavailable; rebuild Rust and C++");
    TORCH_CHECK(scalars.size()>0 && scalars.size()<=256
      && tensors.size()<=512 && tensors.size()==inputs+scalars.size()
      && words.size()==scalars.size()*3,"invalid static graph header");
    std::vector<std::unique_ptr<Argument>> args;
    std::vector<Descriptor> descriptors;
    for(const auto& t:tensors_) {
      validate(t.device());
      TORCH_CHECK(t.layout()==c10::Layout::Strided && t.is_contiguous()
        && t.dim()>=0 && t.dim()<=8 && t.numel()<=std::numeric_limits<uint32_t>::max(),
        "static graph requires contiguous rank 0..8 tensors");
      for(const auto size:t.sizes()) TORCH_CHECK(size<=std::numeric_limits<uint32_t>::max(),
        "static graph dimension exceeds 32-bit kernel indexing");
      const uint32_t dtype=dtype_code(t.scalar_type());
      TORCH_CHECK(dtype<=8 || dtype==12 || dtype==13 || dtype==15,"unsupported static graph storage dtype");
      TORCH_CHECK(!t.requires_grad(),"static graph is inference-only");
      args.emplace_back(std::make_unique<Argument>(t));
      descriptors.push_back(args.back()->desc);
      snapshots_.push_back({args.back()->desc.allocation,t.data_ptr(),args.back()->desc.offset_bytes,
                            t.scalar_type(),t.sizes().vec(),t.strides().vec()});
    }
    std::vector<StaticNode> nodes;
    for(size_t i=0;i<scalars.size();++i) {
      const uint32_t op=words[i*3], a=words[i*3+1], b=words[i*3+2];
      const size_t out=inputs+i;
      const bool root=(op==160 || op==161 || op==172) && a==std::numeric_limits<uint32_t>::max();
      TORCH_CHECK(root || a<out,"static graph node refers to a future input");
      const auto& x=tensors_[root ? out : a]; const auto& y=tensors_[out];
      TORCH_CHECK((dtype_code(x.scalar_type())<=8 && dtype_code(y.scalar_type())<=8)
        || op==0 || op==218 || (op>=112 && op<=115),"FP8 graph storage requires explicit cast or layout operations");
      TORCH_CHECK((op==7 || op==30 || op==173 || op==183 || op==184 || (op>=211 && op<=217) || (op>=189 && op<=200) || (op>=203 && op<=209) || op==57 || op==104 || op==105 || (op>=112 && op<=124) || (op>=134 && op<=172) || x.sizes()==y.sizes())
        && (op==114 || op==172 || op==218 || (op>=147 && op<=161) || (op>=164 && op<=167) || x.scalar_type()==y.scalar_type()),"static graph output mismatch");
      TORCH_CHECK((dtype_code(x.scalar_type())<=2 && dtype_code(y.scalar_type())<=2)
        || op==0 || op==218 || (op>=112 && op<=115) || (op>=134 && op<=136) || (op>=140 && op<=172) || op==193 || (op>=212 && op<=216),
        "nonfloating graph tensors require storage/layout/indexing operations");
      if(op==218) {
        const auto dtype=dtype_code(x.scalar_type());
        TORCH_CHECK(b<out && (dtype<=2 || dtype==12 || dtype==13 || dtype==15)
          && y.scalar_type()==at::kFloat && x.dim()==2 && x.numel()>0
          && std::isfinite(scalars[i]) && scalars[i]>=1 && scalars[i]<=16777216
          && std::floor(scalars[i])==scalars[i],"invalid block-scaled cast input or row block");
        const auto& scales=tensors_[b];const int64_t block=int64_t(scalars[i]);
        TORCH_CHECK(scales.scalar_type()==at::kFloat && x.size(0)%block==0 && scales.numel()>0
          && scales.numel()%(x.size(0)/block)==0 && x.size(1)%(scales.numel()/(x.size(0)/block))==0,
          "invalid block-scaled cast scale grid");
      } else if(op>=212 && op<=216) {
        TORCH_CHECK(b<out && std::isfinite(scalars[i]) && std::floor(scalars[i])==scalars[i],"invalid strided transform input/scalar");
        const auto& right=tensors_[b];
        TORCH_CHECK(x.scalar_type()==right.scalar_type() && (op==214 || a==b),"invalid strided transform operands");
        if(op<=214) {
          TORCH_CHECK(scalars[i]>=0 && scalars[i]<16777216,"invalid strided slice encoding");
          const uint32_t code=uint32_t(scalars[i]),axis=code&7,start=(code>>3)&2047,step=code>>14;
          const auto& full=op==213?y:x;const auto& part=op==213?x:op==214?right:y;
          TORCH_CHECK(step>0 && axis<full.dim() && part.dim()==full.dim()
            && (op!=214 || x.sizes()==y.sizes()) && start<=full.size(axis),"invalid strided slice rank/axis");
          for(int d=0;d<full.dim();++d) TORCH_CHECK(d==axis || full.size(d)==part.size(d),"invalid strided slice shape");
          TORCH_CHECK(part.size(axis)==0 || uint64_t(start)+uint64_t(part.size(axis)-1)*step<uint64_t(full.size(axis)),"strided slice out of bounds");
        } else if(op==215) {
          TORCH_CHECK(scalars[i]==0.0f && !std::signbit(scalars[i]) && y.dim()>=x.dim(),"invalid repeat rank/scalar");
          for(int d=0;d<x.dim();++d) {
            const auto old=x.size(d),size=y.size(d+y.dim()-x.dim());
            TORCH_CHECK(old==0 ? size==0 : size%old==0,"invalid repeat shape");
          }
        } else if(scalars[i]==-1.0f) {
          TORCH_CHECK(y.dim()==1 && (x.numel()==0 ? y.numel()==0 : y.numel()%x.numel()==0),"invalid flat repeat_interleave shape");
        } else {
          TORCH_CHECK(scalars[i]>=0 && scalars[i]<x.dim() && y.dim()==x.dim(),"invalid repeat_interleave axis/rank");
          const int axis=int(scalars[i]);
          for(int d=0;d<x.dim();++d) TORCH_CHECK(d!=axis ? x.size(d)==y.size(d)
            : x.size(d)==0 ? y.size(d)==0 : y.size(d)%x.size(d)==0,"invalid repeat_interleave shape");
        }
      } else if(op==211 || op==217) {
        TORCH_CHECK(b<out && x.dim()>=2 && tensors_[b].dim()>=2 && y.dim()==3
          && x.scalar_type()==tensors_[b].scalar_type() && std::isfinite(scalars[i])
          && scalars[i]>=0 && scalars[i]<(op==217?16777216:4096) && std::floor(scalars[i])==scalars[i],"invalid grouped BMM inputs/axes");
        const uint32_t code=uint32_t(scalars[i]);int axes[4];
        const auto& right=tensors_[b];
        if(op==217) {
          auto factorial=[](int rank) {uint32_t result=1;for(int d=2;d<=rank;++d)result*=d;return result;};
          const uint32_t variants=factorial(int(x.dim()))*(x.dim()-1);
          const uint32_t layouts[2]={code%variants,code/variants};
          const int ranks[2]={int(x.dim()),int(right.dim())};
          for(int side=0;side<2;++side) {
            const int rank=ranks[side];
            TORCH_CHECK(layouts[side]<factorial(rank)*(rank-1),"invalid BMM layout encoding");
            uint32_t permutation=layouts[side]/(rank-1);std::vector<int> available;
            for(int d=0;d<rank;++d)available.push_back(d);
            for(int d=0;d<rank;++d) {
              const uint32_t block=factorial(rank-d-1),position=permutation/block;permutation%=block;
              const int axis=available[position];available.erase(available.begin()+position);
              if(d>=rank-2)axes[side*2+d-rank+2]=axis;
            }
          }
        } else {for(int d=0;d<4;++d)axes[d]=(code>>(3*d))&7;}
        TORCH_CHECK(axes[0]!=axes[1] && axes[2]!=axes[3] && axes[0]<x.dim() && axes[1]<x.dim()
          && axes[2]<right.dim() && axes[3]<right.dim() && x.numel()>0 && right.numel()>0 && y.numel()>0,"invalid grouped BMM dimensions");
        int64_t ab=1,bb=1;
        for(int d=0;d<x.dim();++d)if(d!=axes[0] && d!=axes[1])ab*=x.size(d);
        for(int d=0;d<right.dim();++d)if(d!=axes[2] && d!=axes[3])bb*=right.size(d);
        TORCH_CHECK(x.size(axes[1])==right.size(axes[2]) && y.size(1)==x.size(axes[0])
          && y.size(2)==right.size(axes[3]) && y.size(0)%ab==0 && y.size(0)%bb==0,"invalid grouped BMM dimensions/repeats");
        TORCH_CHECK(uint64_t(y.size(0))<=std::numeric_limits<uint32_t>::max()/(uint64_t(x.size(axes[0]))*x.size(axes[1]))
          && uint64_t(y.size(0))<=std::numeric_limits<uint32_t>::max()/(uint64_t(right.size(axes[2]))*right.size(axes[3])),"grouped BMM indexing limit");
      } else if(op==210) {
        TORCH_CHECK(b<out && x.dim()==2 && x.size(1)>0,"invalid group mask input");
        const auto& groups=tensors_[b];const auto dtype=dtype_code(x.scalar_type());
        TORCH_CHECK(dtype_code(groups.scalar_type())<=8 && groups.dim()==2 && groups.size(0)==x.size(0)
          && groups.size(1)>0 && x.size(1)%groups.size(1)==0,"invalid group indicator shape/dtype");
        TORCH_CHECK(dtype==0 || !std::isfinite(scalars[i]) || std::abs(scalars[i])<=(dtype==1?65504.0f:3.38953139e38f),"group fill exceeds storage dtype");
      } else if(op==207) {
        TORCH_CHECK(b<out && x.dim()==3 && tensors_[b].dim()==3 && y.dim()==3
          && x.scalar_type()==tensors_[b].scalar_type() && std::isfinite(scalars[i])
          && scalars[i]>=0 && scalars[i]<262144 && std::floor(scalars[i])==scalars[i],"invalid permuted BMM inputs/axes");
        const uint32_t code=uint32_t(scalars[i]);int axes[2][3];
        for(int side=0;side<2;++side) {
          unsigned seen=0;for(int d=0;d<3;++d) {
            const int axis=(code>>(side*9+d*3))&7;TORCH_CHECK(axis<3 && !(seen&(1u<<axis)),"invalid BMM permutation");
            seen|=1u<<axis;axes[side][d]=axis;
          }
        }
        const auto& right=tensors_[b];
        TORCH_CHECK(x.size(axes[0][0])==right.size(axes[1][0]) && x.size(axes[0][2])==right.size(axes[1][1])
          && y.size(0)==x.size(axes[0][0]) && y.size(1)==x.size(axes[0][1]) && y.size(2)==right.size(axes[1][2]),"invalid permuted BMM dimensions");
      } else if(op>=203 && op<=206) {
        TORCH_CHECK(b<out,"future product reduction input");
        const auto& right=tensors_[b];const int rank=std::max(x.dim(),right.dim());
        TORCH_CHECK(rank>0 && right.scalar_type()==x.scalar_type() && std::isfinite(scalars[i])
          && scalars[i]>0 && scalars[i]<(1u<<rank) && std::floor(scalars[i])==scalars[i],"invalid product reduction dtype/mask");
        const uint32_t mask=uint32_t(scalars[i]);uint64_t length=1;std::vector<int64_t> expected;
        for(int d=0;d<rank;++d) {
          const int64_t a_size=d<rank-x.dim()?1:x.size(d-rank+x.dim());
          const int64_t b_size=d<rank-right.dim()?1:right.size(d-rank+right.dim());
          TORCH_CHECK(a_size>0 && b_size>0 && (a_size==1 || b_size==1 || a_size==b_size),"invalid product broadcast dimensions");
          const int64_t size=std::max(a_size,b_size);
          TORCH_CHECK(length<=std::numeric_limits<uint32_t>::max()/uint64_t(size),"product indexing overflow");length*=size;
          if(!(mask&(1u<<d)))expected.push_back(size);else if(op==203 || op==205)expected.push_back(1);
        }
        TORCH_CHECK(y.sizes()==c10::IntArrayRef(expected),"invalid product reduction output shape");
      } else if(op==201 || op==202) {
        TORCH_CHECK(x.dim()>0 && x.size(-1)>0 && x.size(-1)%2==0 && b<out
          && std::isfinite(scalars[i]) && scalars[i]>=0 && scalars[i]<out
          && std::floor(scalars[i])==scalars[i],"invalid rotary axis or coefficients");
        const auto& cosine=tensors_[b];const auto& sine=tensors_[size_t(scalars[i])];
        for(const auto* coefficient:{&cosine,&sine}) {
          TORCH_CHECK(coefficient->scalar_type()==x.scalar_type() && coefficient->dim()<=x.dim(),"invalid rotary coefficient dtype/rank");
          for(int d=0;d<coefficient->dim();++d) TORCH_CHECK(coefficient->size(d)==1
            || coefficient->size(d)==x.size(d+x.dim()-coefficient->dim()),"invalid rotary coefficient broadcast");
        }
      } else if(op==194 || op==195) {
        TORCH_CHECK(x.scalar_type()==at::kFloat && x.dim()>=1 && x.dim()<=7 && x.size(-1)>0
          && b==a && scalars[i]==0.0f && !std::signbit(scalars[i]),"invalid segment scan input");
        auto expected=x.sizes().vec();expected.push_back(x.size(-1));
        TORCH_CHECK(y.sizes()==c10::IntArrayRef(expected),"invalid segment scan output shape");
      } else if(op==193) {
        TORCH_CHECK(x.scalar_type()==at::kByte && x.dim()>0 && x.size(-1)%2==0 && x.dim()==y.dim()
          && b==a && scalars[i]==0.0f && !std::signbit(scalars[i]),"invalid byte packing input");
        for(int d=0;d<x.dim();++d) TORCH_CHECK(y.size(d)==(d+1==x.dim()?x.size(d)/2:x.size(d)),"invalid byte packing output shape");
      } else if((op>=189 && op<=192) || op==208 || op==209) {
        TORCH_CHECK(b==a && std::isfinite(scalars[i]) && scalars[i]>0 && scalars[i]<(1u<<x.dim())
          && std::floor(scalars[i])==scalars[i],"invalid extremum reduction mask");
        const auto mask=static_cast<uint32_t>(scalars[i]);int out_dim=0;
        for(int d=0;d<x.dim();++d) {
          const bool reduced=(mask&(1u<<d))!=0;
          TORCH_CHECK(!reduced || x.size(d)>0,"extremum reduction has no identity for an empty axis");
          if(!reduced || op<=190 || op==208) {
            TORCH_CHECK(out_dim<y.dim() && y.size(out_dim)==(reduced ? 1 : x.size(d)),"invalid extremum output shape");++out_dim;
          }
        }
        TORCH_CHECK(out_dim==y.dim(),"invalid extremum output rank");
      } else if(op>=174 && op<=188) {
        TORCH_CHECK(b<out && std::isfinite(scalars[i]),"invalid graph activation operand/scalar");
        const auto& r=tensors_[b];
        TORCH_CHECK(x.scalar_type()==r.scalar_type(),"activation operand dtype mismatch");
        if(op==183 || op==184) {
          const int rank=std::max(x.dim(),r.dim());
          TORCH_CHECK(y.dim()==rank,"activation broadcast rank mismatch");
          for(int d=0;d<rank;++d) {
            const auto lhs=d<rank-x.dim() ? 1 : x.size(d-rank+x.dim());
            const auto rhs=d<rank-r.dim() ? 1 : r.size(d-rank+r.dim());
            TORCH_CHECK((lhs==rhs || lhs==1 || rhs==1) && y.size(d)==(lhs==1 ? rhs : lhs),"activation broadcast shape mismatch");
          }
        } else if(op==176 || op==177 || op==179 || op==186 || op==188) {
          TORCH_CHECK(x.sizes()==r.sizes(),"activation gradient shape mismatch");
        } else {TORCH_CHECK(b==a,"unary activation requires a dummy operand");}
        TORCH_CHECK((op>=178 && op<=181) || op==187 || op==188 || (scalars[i]==0.0f && !std::signbit(scalars[i])),"invalid activation scalar");
      } else if(op==172) {
        float step;std::memcpy(&step,&b,sizeof(step));
        TORCH_CHECK(dtype_code(y.scalar_type())<=2 && y.dim()==1 && std::isfinite(scalars[i]) && std::isfinite(step),
          "invalid floating arange graph parameters");
      } else if(op>=162 && op<=171) {
        TORCH_CHECK(b<out && std::isfinite(scalars[i]),"invalid graph mask operand/scalar");
        if(op<=163) {
          TORCH_CHECK(b==a && x.dim()>=2 && x.sizes()==y.sizes() && std::floor(scalars[i])==scalars[i]
            && double(scalars[i])>=-2147483648.0 && double(scalars[i])<=2147483647.0,"invalid graph triangular mask");
        } else {
          const auto& r=tensors_[b];const int rank=std::max(x.dim(),r.dim());
          TORCH_CHECK(scalars[i]==0.0f && x.scalar_type()==r.scalar_type() && y.dim()==rank
            && ((op!=164 && op!=168) || b==a),"invalid logical/bitwise graph operands");
          for(int d=0;d<rank;++d) {
            const int64_t lhs=d<rank-x.dim() ? 1 : x.size(d-rank+x.dim());
            const int64_t rhs=d<rank-r.dim() ? 1 : r.size(d-rank+r.dim());
            TORCH_CHECK((lhs==rhs || lhs==1 || rhs==1) && y.size(d)==(lhs==1 ? rhs : lhs),"invalid logical/bitwise graph broadcast");
          }
          TORCH_CHECK(op<=167 ? y.scalar_type()==at::kBool : dtype_code(x.scalar_type())>=3,"invalid logical/bitwise graph dtype");
        }
      } else if(op==160 || op==161) {
        const double start=scalars[i];const uint32_t dtype=dtype_code(y.scalar_type());
        double low=0,high=0;
        if(dtype>=4) {
          if(dtype==4){low=-9223372036854775808.0;high=9223372036854775808.0;}
          else if(dtype==5){low=-2147483648.0;high=2147483647.0;}
          else if(dtype==6){low=-32768.0;high=32767.0;}
          else if(dtype==7){low=-128.0;high=127.0;}
          else {low=0.0;high=255.0;}
          TORCH_CHECK(std::isfinite(start) && std::floor(start)==start && start>=low
            && (dtype==4 ? start<high : start<=high),"factory scalar exceeds integer dtype");
        } else if(dtype==1 || dtype==2) {
          TORCH_CHECK(!std::isfinite(start) || std::abs(start)<=(dtype==1 ? 65504.0 : double(3.38953139e38f)),"factory scalar exceeds floating dtype");
        }
        if(op==160) {TORCH_CHECK(b==a,"full requires one device anchor");}
        else {
          TORCH_CHECK(dtype>=4 && y.dim()==1 && b!=0,"arange_integer requires integer vector and nonzero signed step");
          const int64_t step=b<=uint32_t(std::numeric_limits<int32_t>::max()) ? int64_t(b) : int64_t(b)-4294967296LL;
          const int64_t first=int64_t(start),offset=std::max<int64_t>(y.numel()-1,0)*step;
          TORCH_CHECK(offset>=0 ? first<=std::numeric_limits<int64_t>::max()-offset
            : first>=std::numeric_limits<int64_t>::min()-offset,"arange_integer endpoint overflows int64");
          const int64_t end=first+offset;
          TORCH_CHECK(dtype==4 || (end>=int64_t(low) && end<=int64_t(high)),"arange_integer endpoint exceeds output dtype");
        }
      } else if(op>=146 && op<=159) {
        TORCH_CHECK(b<out,"predicate graph input must precede output");
        const auto& right=tensors_[b];
        auto broadcast=[&](std::initializer_list<const at::Tensor*> values) {
          int rank=0;for(auto value:values)rank=std::max(rank,int(value->dim()));
          if(y.dim()!=rank)return false;
          for(int d=0;d<rank;++d) {
            int64_t size=1;
            for(auto value:values) {
              const int64_t next=d<rank-value->dim() ? 1 : value->size(d-rank+value->dim());
              if(size!=1 && next!=1 && size!=next)return false;if(size==1)size=next;
            }
            if(y.size(d)!=size)return false;
          }
          return true;
        };
        auto scalar_fits=[&]() {
          const double value=scalars[i];const uint32_t dtype=dtype_code(x.scalar_type());
          if(dtype==0 || dtype==3)return true;
          if(dtype==1)return !std::isfinite(value) || std::abs(value)<=65504.0;
          if(dtype==2)return !std::isfinite(value) || std::abs(value)<=double(3.38953139e38f);
          if(!std::isfinite(value) || std::floor(value)!=value)return false;
          if(dtype==4)return value>=-9223372036854775808.0 && value<9223372036854775808.0;
          if(dtype==5)return value>=-2147483648.0 && value<=2147483647.0;
          if(dtype==6)return value>=-32768.0 && value<=32767.0;
          if(dtype==7)return value>=-128.0 && value<=127.0;
          return dtype==8 && value>=0.0 && value<=255.0;
        };
        if(op==146) {
          TORCH_CHECK(right.scalar_type()==at::kBool && scalar_fits() && broadcast({&x,&right}),"invalid masked_fill graph inputs/scalar");
        } else if(op==147) {
          TORCH_CHECK(std::isfinite(scalars[i]) && scalars[i]>=0 && scalars[i]<out && std::floor(scalars[i])==scalars[i],"where third input must precede output");
          const auto& third=tensors_[size_t(scalars[i])];
          TORCH_CHECK(x.scalar_type()==at::kBool && right.scalar_type()==third.scalar_type() && y.scalar_type()==right.scalar_type()
            && broadcast({&x,&right,&third}),"invalid where graph dtype/broadcast");
        } else {
          TORCH_CHECK(y.scalar_type()==at::kBool,"comparison graph output must be bool");
          if(op<154) {TORCH_CHECK(scalars[i]==0.0f && !std::signbit(scalars[i]) && x.scalar_type()==right.scalar_type()
            && broadcast({&x,&right}),"invalid graph comparison inputs");}
          else {TORCH_CHECK(b==a && x.sizes()==y.sizes() && scalar_fits()
            && (x.scalar_type()!=at::kBool || scalars[i]==0.0f || scalars[i]==1.0f),"invalid scalar comparison");}
        }
      } else if(op>=140 && op<=145) {
        TORCH_CHECK(b<out && std::isfinite(scalars[i]) && scalars[i]>=0 && scalars[i]<16777216.0f
          && std::floor(scalars[i])==scalars[i] && tensors_[b].scalar_type()==x.scalar_type(),"invalid transform graph parameter/input");
        const auto code=uint32_t(scalars[i]);const int axis=op==142 ? int(code) : int(code&7u);const int64_t start=code>>3;
        const auto& right=tensors_[b];
        if(op==142) {
          TORCH_CHECK(axis<x.dim() && right.dim()==x.dim() && y.dim()==x.dim(),"invalid cat rank/axis");
          for(int d=0;d<x.dim();++d) TORCH_CHECK(d==axis ? y.size(d)==x.size(d)+right.size(d) : (x.size(d)==right.size(d) && x.size(d)==y.size(d)),"invalid cat dimensions");
        } else {
          const auto& full=(op==143 || op==144) ? y : x;
          const auto& part=(op==143 || op==144) ? x : (op==145 ? right : y);
          TORCH_CHECK(axis<full.dim() && (op==145 ? x.sizes()==y.sizes() : b==a),"invalid slice/select operands");
          auto expected=full.sizes().vec();
          if(op==141 || op==144) {
            TORCH_CHECK(start<full.size(axis),"graph select out of bounds");expected.erase(expected.begin()+axis);
          } else {
            TORCH_CHECK(part.dim()==full.dim() && start+part.size(axis)<=full.size(axis),"graph slice out of bounds");expected[axis]=part.size(axis);
          }
          TORCH_CHECK(part.sizes()==c10::IntArrayRef(expected),"slice/select output mismatch");
        }
      } else if(op>=134 && op<=139) {
        TORCH_CHECK(b<out && std::isfinite(scalars[i]) && std::floor(scalars[i])==scalars[i],"invalid graph indexing parameter/input");
        const auto& indices=tensors_[b];
        TORCH_CHECK(indices.scalar_type()==at::kLong || indices.scalar_type()==at::kInt,"graph indices require int32/int64");
        if(op>=136 && op<=138) {
          const auto& weight=op==136 ? x : y; const auto& gradient=op==136 ? y : x;
          TORCH_CHECK(weight.dim()==2,"embedding weights must be a matrix");
          auto expected=indices.sizes().vec();expected.push_back(weight.size(1));
          TORCH_CHECK(gradient.sizes()==c10::IntArrayRef(expected)
            && (op==136 ? scalars[i]==0.0f : (scalars[i]>=-1.0f && scalars[i]<weight.size(0))),"embedding graph shape/padding mismatch");
        } else {
          const auto& source=op==139 ? y : x; const auto& indexed=op==139 ? x : y;
          TORCH_CHECK(scalars[i]>=0 && scalars[i]<source.dim() && indexed.dim()==source.dim(),"invalid indexing axis/rank");
          const int axis=int(scalars[i]);
          if(op==134) {
            TORCH_CHECK(indices.dim()<=1,"index_select indices must be scalar/vector");
            for(int d=0;d<source.dim();++d) TORCH_CHECK(indexed.size(d)==(d==axis ? indices.numel() : source.size(d)),"index_select output mismatch");
          } else {
            TORCH_CHECK(indices.sizes()==indexed.sizes(),"gather indices/output mismatch");
            for(int d=0;d<source.dim();++d) TORCH_CHECK(d==axis || indices.size(d)<=source.size(d),"gather shape exceeds source");
          }
        }
      } else if(op==112 || op==114) {
        TORCH_CHECK(b==a && scalars[i]==0.0f && !std::signbit(scalars[i])
          && (op==112 ? x.numel()==y.numel() : x.sizes()==y.sizes()),"invalid graph reshape/cast");
      } else if(op==113) {
        TORCH_CHECK(b==a && x.dim()==y.dim() && std::isfinite(scalars[i]) && scalars[i]>=0
          && std::floor(scalars[i])==scalars[i] && scalars[i]<float(1u<<(3*x.dim())),"invalid graph permutation encoding");
        const auto code=static_cast<uint32_t>(scalars[i]); uint32_t seen=0;
        for(int d=0;d<x.dim();++d) {
          const uint32_t axis=(code>>(3*d))&7u;
          TORCH_CHECK(axis<uint32_t(x.dim()) && !(seen&(1u<<axis)) && y.size(d)==x.size(axis),"invalid graph permutation axes/shape");
          seen|=1u<<axis;
        }
      } else if(op==115) {
        TORCH_CHECK(b==a && scalars[i]==0.0f && !std::signbit(scalars[i]) && y.dim()>=x.dim(),"invalid graph expansion");
        for(int d=0;d<x.dim();++d) TORCH_CHECK(x.size(d)==1 || x.size(d)==y.size(y.dim()-x.dim()+d),"invalid graph expansion shape");
      } else if(op==57 || (op>=116 && op<=118) || (op>=121 && op<=124)) {
        TORCH_CHECK(b<out && tensors_[b].scalar_type()==x.scalar_type() && std::isfinite(scalars[i])
          && (op==116 || (scalars[i]==0.0f && !std::signbit(scalars[i]))),"invalid broadcast graph inputs");
        const auto& r=tensors_[b]; const int rank=std::max(x.dim(),r.dim());
        TORCH_CHECK(y.dim()==rank,"invalid broadcast output rank");
        for(int d=0;d<rank;++d) {
          const auto lhs=d<rank-x.dim() ? 1 : x.size(d-rank+x.dim());
          const auto rhs=d<rank-r.dim() ? 1 : r.size(d-rank+r.dim());
          TORCH_CHECK((lhs==rhs || lhs==1 || rhs==1) && y.size(d)==(lhs==1 ? rhs : lhs),"invalid broadcast shape");
        }
      } else if(op==119 || op==120) {
        TORCH_CHECK(b==a && std::isfinite(scalars[i]) && scalars[i]>0 && scalars[i]<(1u<<x.dim())
          && std::floor(scalars[i])==scalars[i],"invalid squeezed reduction mask");
        const auto mask=static_cast<uint32_t>(scalars[i]); int out_dim=0;
        for(int d=0;d<x.dim();++d) if(!(mask&(1u<<d))) {
          TORCH_CHECK(out_dim<y.dim() && y.size(out_dim)==x.size(d),"invalid squeezed reduction shape"); ++out_dim;
        }
        TORCH_CHECK(out_dim==y.dim(),"invalid squeezed reduction rank");
      } else if(op==0 || op==3 || (op>=9 && op<=14) || op==17 || (op>=19 && op<=25) || op==27 || (op>=35 && op<=40) || op==43 || op==45 || op==108) {
        TORCH_CHECK(b==a && scalars[i]==0.0f && !std::signbit(scalars[i]),"invalid unary static graph node");
      } else if(op==1 || op==2 || op==8 || op==15 || op==16 || op==18 || op==101) {
        TORCH_CHECK(b<out && tensors_[b].sizes()==x.sizes() && tensors_[b].scalar_type()==x.scalar_type(),
                    "static graph broadcasting/type promotion is not supported");
        TORCH_CHECK(std::isfinite(scalars[i]) && (op==1 || (scalars[i]==0.0f && !std::signbit(scalars[i]))),
                    "invalid static graph scalar");
      } else if((op>=109 && op<=111) || (op>=129 && op<=133)) {
        TORCH_CHECK(b==a && std::isfinite(scalars[i]), "invalid scalar graph operation");
      } else if(op==7 || op==30 || op==173 || (op>=196 && op<=200)) {
        TORCH_CHECK(b<out, "future matrix graph input");
        const auto& r=tensors_[b]; const int rank=(op==30 || (op>=196 && op<=198)) ? 3 : 2;
        const bool lhs_transposed=op==197 || op==198 || op==199 || op==200;
        const bool rhs_transposed=op==173 || op==196 || op==198 || op==200;
        TORCH_CHECK(x.dim()==rank && r.dim()==rank && y.dim()==rank && x.scalar_type()==r.scalar_type()
          && x.size(lhs_transposed ? -2 : -1)==r.size(rhs_transposed ? -1 : -2)
          && y.size(-2)==x.size(lhs_transposed ? -1 : -2) && y.size(-1)==r.size(rhs_transposed ? -2 : -1)
          && (rank==2 || (x.size(0)==r.size(0) && y.size(0)==x.size(0)))
          && scalars[i]==0.0f && !std::signbit(scalars[i]), "invalid graph matrix multiply");
      } else if(op==102 || op==103 || op==106 || op==107) {
        TORCH_CHECK(std::isfinite(scalars[i]) && scalars[i]>=0 && scalars[i]<x.dim()
          && std::floor(scalars[i])==scalars[i] && b<out, "invalid graph softmax axis/input");
        TORCH_CHECK(tensors_[b].sizes()==x.sizes() && tensors_[b].scalar_type()==x.scalar_type()
          && (op>=106 || b==a), "graph softmax gradient/output mismatch");
      } else if(op==104 || op==105) {
        TORCH_CHECK(std::isfinite(scalars[i]) && scalars[i]>0 && scalars[i]<(1u<<x.dim())
          && std::floor(scalars[i])==scalars[i] && b==a && y.dim()==x.dim(), "invalid graph reduction mask");
        const auto mask=static_cast<uint32_t>(scalars[i]);
        for(int d=0;d<x.dim();++d) TORCH_CHECK(y.size(d)==((mask & (1u<<d)) ? 1 : x.size(d)),
          "graph reduction output mismatch");
      } else if(op==100) {
        TORCH_CHECK(x.dim()>0 && std::isfinite(scalars[i]) && scalars[i]>0.0f,"invalid RMSNorm epsilon/rank");
        TORCH_CHECK(x.numel()==0 || x.numel()/x.size(-1)<=std::numeric_limits<uint32_t>::max()/32,
                    "RMSNorm row grid overflow");
        if(b!=std::numeric_limits<uint32_t>::max()) {
          TORCH_CHECK(b<out && tensors_[b].dim()==1 && tensors_[b].size(0)==x.size(-1)
            && tensors_[b].scalar_type()==x.scalar_type(),"invalid RMSNorm weight");
        }
      } else { TORCH_CHECK(false,"unsupported static graph operator; no CPU fallback"); }
      nodes.push_back({op,a,b,scalars[i]});
    }
    // Inclusive lifetimes: never overwrite an input used by the SAME node.
    // Inputs never share writable storage. Leaf outputs remain live until end.
    // Python additionally dedicates storage to every externally requested result.
    std::vector<size_t> last(tensors_.size());
    std::vector<bool> consumed(tensors_.size(),false);
    for(size_t i=0;i<last.size();++i)last[i]=i;
    for(size_t i=0;i<nodes.size();++i) {
      const auto& n=nodes[i];const size_t out=inputs+i;
      if(n.a!=std::numeric_limits<uint32_t>::max()) {last[n.a]=std::max(last[n.a],out);consumed[n.a]=true;}
      if(n.op!=161 && n.op!=172 && n.b!=std::numeric_limits<uint32_t>::max()) {
        last[n.b]=std::max(last[n.b],out);consumed[n.b]=true;
      }
      if(n.op==147 || n.op==201 || n.op==202) {
        const size_t third=size_t(n.scalar);last[third]=std::max(last[third],out);consumed[third]=true;
      }
    }
    for(size_t i=0;i<tensors_.size();++i)
      if(i<inputs || !consumed[i])last[i]=tensors_.size();
    for(size_t i=inputs;i<tensors_.size();++i) {
      for(size_t j=0;j<i;++j) {
        if(tensors_[i].storage().unsafeGetStorageImpl()==tensors_[j].storage().unsafeGetStorageImpl()) {
          const auto& a=snapshots_[i];const auto& b=snapshots_[j];
          TORCH_CHECK(j>=inputs && last[j]<i && a.allocation==b.allocation
              && a.address==b.address && a.offset==b.offset && a.dtype==b.dtype && a.shape==b.shape
              && a.strides==b.strides,
              "static graph writable slots need distinct storage or nonoverlapping lifetimes with exact buffer reuse");
        } else {
          at::assert_no_overlap(tensors_[i],tensors_[j]);
        }
      }
    }
    stream_=stream_command(0);
    check(static_graph_native(0,&plan_,descriptors.data(),descriptors.size(),nodes.data(),nodes.size(),
                              inputs,static_cast<uint32_t>(tracked)|(static_cast<uint32_t>(infer)<<1),&edges_));
    TORCH_CHECK(plan_,"native graph build returned a null plan");
  }
  ~StaticGraphBridge() {
    if(plan_) {
      uint64_t ignored=0;
      if(static_graph_native(4,&plan_,nullptr,0,nullptr,0,0,0,&ignored))
        std::fprintf(stderr,"RUDA static graph cleanup failed (native resources retained): %s\n",error_native());
    }
  }
  void run(bool eager=false) {
    std::lock_guard<std::mutex> lock(mutex_); check_bindings();
    // Account for raw native writes in PyTorch's tensor mutation tracking.
    for(size_t i=inputs_;i<tensors_.size();++i) tensors_[i].unsafeGetTensorImpl()->bump_version();
    command(eager?5:1);
  }
  void synchronize() {std::lock_guard<std::mutex> lock(mutex_);TORCH_CHECK(plan_,"static graph is closed");command(2);}
  bool query() {std::lock_guard<std::mutex> lock(mutex_);TORCH_CHECK(plan_,"static graph is closed");return command(3)!=0;}
  bool query_completion() {std::lock_guard<std::mutex> lock(mutex_);TORCH_CHECK(plan_ && tracked_,"tracked graph required");return command(6)!=0;}
  void wait_completion() {std::lock_guard<std::mutex> lock(mutex_);TORCH_CHECK(plan_ && tracked_,"tracked graph required");command(7);}
  void close() {std::lock_guard<std::mutex> lock(mutex_);if(plan_)command(4);tensors_.clear();snapshots_.clear();}
  uint64_t edge_count() const {return edges_;}
  uint64_t stream_id() const {return stream_;}
};
