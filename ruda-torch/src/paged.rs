//! In-process bridge to the public ruDNN paged kernel. No standalone PTX runtime.
use super::*;
use rudnn::paged_attention::{DevicePlan,HostPlan,SplitWorkspace,MAX_SPLITS,OrderedBackwardWorkspace};
use std::sync::Mutex;

pub(super) static SPLIT_CALLS: AtomicU64=AtomicU64::new(0);
pub(super) static WORKSPACE_ALLOCS: AtomicU64=AtomicU64::new(0);
pub(super) static WORKSPACE_BYTES: AtomicU64=AtomicU64::new(0);

pub(super) static BACKWARD_CALLS: AtomicU64=AtomicU64::new(0);
pub(super) static BACKWARD_HISTORY_BYTES: AtomicU64=AtomicU64::new(0);
pub(super) static BACKWARD_PLACEHOLDER_BYTES: AtomicU64=AtomicU64::new(0);

/// Additive capability. op=4 allows null optional gradient descriptors.
#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_paged_backward_api_version()->u32 { 2 }

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_paged_quantized_api_version()->u32 { 1 }

/// Q, Qposition, K triple, V triple, position-cache triple, output.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_paged_quantized(plan:*const NativePlan,
    descriptors:*const Descriptor,count:usize,layout:*const u32,layout_count:usize,scale:f32,causal:bool)->i32 {
    checked(|| {
        use ruda_kernel::{quantization::packed::{PackedTensor,PackedFormat},tensor::allocation::empty_device_contiguous_dtype};
        use ruda_core::tensor::{Shape,DType};
        assert!(!plan.is_null() && count==12 && layout_count==10);
        let native=unsafe{&*plan};let layout=unsafe{crate::abi::slice(layout,layout_count)};
        let ds=unsafe{crate::abi::slice(descriptors,count)};
        let views:Vec<_>=ds.iter().map(|d|unsafe{crate::abi::view(d)}).collect();
        let tensors:Vec<_>=views.iter().map(primitives::tensor).collect();let q=&tensors[0];
        let empty=empty_device_contiguous_dtype(q.client.clone(),q.device.clone(),Shape::new([0]),DType::I32);
        let cache=|start:usize,format:u32,group:u32,width:u32| PackedTensor::new(
            tensors[start].clone(),tensors[start+1].clone(),tensors[start+2].clone(),empty.clone(),
            Shape::new([native.plan.host().page_count(),native.plan.host().page_size(),layout[3] as usize,width as usize]),
            PackedFormat::from_code(format),group as usize);
        let k=cache(2,layout[0],layout[1],layout[2]);
        let v=cache(5,layout[4],layout[5],layout[6]);
        let kp=(layout[9]>0).then(||cache(8,layout[7],layout[8],layout[9]));
        let position=kp.as_ref().map(|kp|(&tensors[1],kp));
        let mut guard=native.workspace.lock().expect("paged workspace lock poisoned");
        let workspace=if native.splits>1 && q.meta.num_elements()!=0 {
            assert_eq!(q.meta.num_dims(),3);
            if !guard.as_ref().is_some_and(|w|w.matches(q,q.meta.shape()[0],q.meta.shape()[1],v.width(),native.splits)) {
                *guard=Some(SplitWorkspace::new(q,q.meta.shape()[0],q.meta.shape()[1],v.width(),native.splits)
                    .expect("quantized paged split workspace"));
            }
            guard.as_mut()
        } else {None};
        let output=native.plan.attention_quantized(q,&k,&v,position,scale,causal,workspace).expect("quantized paged attention failed");
        primitives::store(output,&views[11]);
        LAUNCHES.fetch_add(if native.splits>1 {2} else {1},Ordering::Relaxed);
        finish_dispatch(&client());
    })
}

pub(super) static ORDERED_CALLS: AtomicU64=AtomicU64::new(0);
pub(super) static ORDERED_ALLOCS: AtomicU64=AtomicU64::new(0);
pub(super) static ORDERED_BYTES: AtomicU64=AtomicU64::new(0);

pub struct NativePlan {
    plan:DevicePlan<CudaRuntime>,
    splits:usize,
    // Calls are enqueued under one lock, always on the plan's ordered queue.
    // No workspace tensor escapes, so reuse cannot race another caller's merge.
    workspace:Mutex<Option<SplitWorkspace<CudaRuntime>>>,
    ordered:Mutex<Option<OrderedBackwardWorkspace<CudaRuntime>>>,
}

/// op=0: validate/upload six-word ABI9 schedule spec; op=1: attention (1 or 2 kernels);
/// op=2: release metadata; op=3: full backward; op=4: selected atomic backward (API 1); op=5: ordered history backward (API 2). The C++ owner guarantees all descriptor lifetimes,
/// disjoint output storage and exactly one destruction of the opaque plan.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_paged(op:u32, plan:*mut *mut NativePlan,
    q:*const Descriptor,k:*const Descriptor,v:*const Descriptor,
    qp:*const Descriptor,kp:*const Descriptor,out:*const Descriptor,
    grad:*const Descriptor,dq:*const Descriptor,dk:*const Descriptor,dv:*const Descriptor,
    dqp:*const Descriptor,dkp:*const Descriptor,
    words:*const u32,word_count:usize,spec:*const u32,scale:f32,causal:bool)->i32
{
    checked(|| {
        assert!(!plan.is_null());
        match op {
            0=>{
                assert!(unsafe{(*plan).is_null()} && !q.is_null() && !spec.is_null());
                let spec=unsafe{crate::abi::slice(spec,6)};
                assert!(!words.is_null() || word_count==0);
                let words=if word_count==0 { &[] } else {unsafe{crate::abi::slice(words,word_count)}};
                let host=HostPlan::from_packed(spec[0] as usize,spec[1] as usize,
                    spec[2] as usize,spec[3] as usize,spec[4] as usize,words).expect("invalid paged schedule");
                let like=primitives::tensor(&unsafe{crate::abi::view(q)});
                let splits=spec[5] as usize;
                assert!((1..=MAX_SPLITS).contains(&splits),"paged split count must be 1..32");
                let native=Box::new(NativePlan{plan:DevicePlan::upload(host,&like),splits,workspace:Mutex::new(None),ordered:Mutex::new(None)});
                unsafe{*plan=Box::into_raw(native);}
            }
            1=>{
                assert!(!q.is_null() && !k.is_null() && !v.is_null() && !out.is_null());
                assert!(!unsafe{*plan}.is_null());
                let plan=unsafe{&**plan};
                let q=primitives::tensor(&unsafe{crate::abi::view(q)});
                let k=primitives::tensor(&unsafe{crate::abi::view(k)});
                let v=primitives::tensor(&unsafe{crate::abi::view(v)});
                let out=primitives::tensor(&unsafe{crate::abi::view(out)});
                assert_eq!(qp.is_null(),kp.is_null());
                let position=if qp.is_null(){None}else{
                    Some((primitives::tensor(&unsafe{crate::abi::view(qp)}),primitives::tensor(&unsafe{crate::abi::view(kp)})))
                };
                let position=position.as_ref().map(|(a,b)|(a,b));
                if plan.splits>1 && q.meta.num_elements()!=0 {
                    assert_eq!(q.meta.num_dims(),3,"paged query rank must be 3");
                    assert_eq!(v.meta.num_dims(),4,"paged value cache rank must be 4");
                    let (queries,heads,dv)=(q.meta.shape()[0],q.meta.shape()[1],v.meta.shape()[3]);
                    let mut workspace=plan.workspace.lock().expect("paged workspace lock poisoned");
                    let reuse=workspace.as_ref().map(|w| w.matches(&q,queries,heads,dv,plan.splits)).unwrap_or(false);
                    if !reuse {
                        let next=SplitWorkspace::new(&q,queries,heads,dv,plan.splits).expect("invalid/oversized split workspace");
                        WORKSPACE_ALLOCS.fetch_add(1,Ordering::Relaxed);
                        WORKSPACE_BYTES.fetch_add(next.bytes() as u64,Ordering::Relaxed);
                        *workspace=Some(next);
                    }
                    // SAFETY: C++ allocates/checks a disjoint output; the lock
                    // serializes scratch writes and merge on the same queue.
                    unsafe{plan.plan.attention_into_split(&q,&k,&v,position,&out,scale,causal,workspace.as_mut().unwrap())}
                        .expect("RUDA split paged attention failed");
                    SPLIT_CALLS.fetch_add(1,Ordering::Relaxed);
                    if out.meta.num_elements()!=0 {
                        let launches=if workspace.as_ref().unwrap().fused_merge() {1} else {2};
                        LAUNCHES.fetch_add(launches,Ordering::Relaxed);
                    }
                } else {
                    // SAFETY: C++ allocates a fresh output and checks overlap.
                    unsafe{plan.plan.attention_into(&q,&k,&v,position,&out,scale,causal)}
                        .expect("RUDA fused paged attention failed");
                    if out.meta.num_elements()!=0 { LAUNCHES.fetch_add(1,Ordering::Relaxed); }
                }
                finish_dispatch(&client());
            }
            3 | 4 | 5=>{
                assert!(!q.is_null() && !k.is_null() && !v.is_null() && !grad.is_null());
                assert!(!unsafe{*plan}.is_null());
                let native=unsafe{&**plan};
                let q=primitives::tensor(&unsafe{crate::abi::view(q)});
                let k=primitives::tensor(&unsafe{crate::abi::view(k)});
                let v=primitives::tensor(&unsafe{crate::abi::view(v)});
                let grad=primitives::tensor(&unsafe{crate::abi::view(grad)});
                assert_eq!(qp.is_null(),kp.is_null());
                if op==3 {
                    // Preserve the original op=3 ABI contract.
                    assert!(!dq.is_null() && !dk.is_null());
                    if qp.is_null() { assert!(!dv.is_null()); }
                    else { assert!(!dqp.is_null() && !dkp.is_null()); }
                }
                let read_optional=|ptr:*const Descriptor| {
                    if ptr.is_null() { None } else { Some(primitives::tensor(&unsafe{crate::abi::view(ptr)})) }
                };
                let dq=read_optional(dq); let dk=read_optional(dk); let dv=read_optional(dv);
                let dqp=read_optional(dqp); let dkp=read_optional(dkp);
                let ordered_requested=op==5 && q.meta.shape()[0]!=0
                    && (dk.is_some() || dv.is_some() || dkp.is_some());
                // Keep the temporary tensors and same-queue lock alive through
                // both submissions. Drop the guard BEFORE surfacing errors.
                let result=(|| -> Result<rudnn::paged_attention::BackwardReport,rudnn::paged_attention::PagedAttentionError> {
                    let mut ordered=if ordered_requested {
                        Some(native.ordered.lock().expect("ordered paged workspace poisoned"))
                    } else { None };
                    if let Some(guard)=ordered.as_mut() {
                        if !guard.as_ref().map(|w| w.matches(&native.plan,&q)).unwrap_or(false) {
                            let next=OrderedBackwardWorkspace::new(&native.plan,&q)?;
                            ORDERED_ALLOCS.fetch_add(1,Ordering::Relaxed);
                            ORDERED_BYTES.fetch_add(next.bytes() as u64,Ordering::Relaxed);
                            **guard=Some(next);
                        }
                    }
                    if qp.is_null() {
                        assert!(dqp.is_none() && dkp.is_none());
                        if let Some(guard)=ordered.as_mut() {
                            unsafe{native.plan.attention_backward_ordered_into(&q,&k,&v,&grad,
                                dq.as_ref(),dk.as_ref(),dv.as_ref(),scale,causal,guard.as_mut().unwrap())}
                        } else {
                            unsafe{native.plan.attention_backward_selected_into(&q,&k,&v,&grad,
                                dq.as_ref(),dk.as_ref(),dv.as_ref(),scale,causal)}
                        }
                    } else {
                        assert!(dv.is_none());
                        let qp=primitives::tensor(&unsafe{crate::abi::view(qp)});
                        let kp=primitives::tensor(&unsafe{crate::abi::view(kp)});
                        if let Some(guard)=ordered.as_mut() {
                            unsafe{native.plan.mla_backward_ordered_into(&q,&qp,&k,&kp,&grad,
                                dq.as_ref(),dqp.as_ref(),dk.as_ref(),dkp.as_ref(),scale,causal,guard.as_mut().unwrap())}
                        } else {
                            unsafe{native.plan.mla_backward_selected_into(&q,&qp,&k,&kp,&grad,
                                dq.as_ref(),dqp.as_ref(),dk.as_ref(),dkp.as_ref(),scale,causal)}
                        }
                    }
                })();
                let report=result.expect("RUDA selected paged backward failed");
                if report.ordered_history { ORDERED_CALLS.fetch_add(1,Ordering::Relaxed); }
                LAUNCHES.fetch_add(report.kernel_launches,Ordering::Relaxed);
                BACKWARD_CALLS.fetch_add(1,Ordering::Relaxed);
                BACKWARD_HISTORY_BYTES.fetch_add(report.history_workspace_bytes as u64,Ordering::Relaxed);
                BACKWARD_PLACEHOLDER_BYTES.fetch_add(report.placeholder_bytes as u64,Ordering::Relaxed);
                finish_dispatch(&client());
            }
            2=>{if !unsafe{(*plan).is_null()} {unsafe{drop(Box::from_raw(*plan));*plan=std::ptr::null_mut();}}}
            _=>panic!("unknown paged plan operation"),
        }
    })
}
