use super::*;
use super::scientific::address;
use ruda_kernel::dsl::prelude::{Tensor, fma};

#[ruda(fast_math = FastMath::ReducedPrecision.into())]
fn cuda_inverse_sqrt<F:Float>(x:F)->F {x.inverse_sqrt()}

#[ruda]
fn vector_merge(left:f32,left_var:f32,left_n:u32,right:f32,right_var:f32,right_n:u32)->(f32,f32,u32) {
    let n=left_n+right_n;let mut mean=0.0f32;let mut variance=0.0f32;
    if n>0 {
        let coef=1.0f32/f32::cast_from(n);let a=f32::cast_from(right_n)*coef;let b=f32::cast_from(left_n)*coef;let delta=left-right;
        mean=fma(a,right,b*left);variance=fma(delta*delta*f32::cast_from(right_n),b,right_var+left_var);
    }
    (mean,variance,n)
}

#[ruda(launch)]
fn vector_statistics<S:Float+RudaElement>(a:&Tensor<S>,mean:&mut Tensor<f32>,variance:&mut Tensor<f32>,rstd:&mut Tensor<f32>,eps:f32) {
    let row=RUDA_POS_X as usize;let lane=UNIT_POS as usize;let width=a.shape(1);
    let mut sm=SharedMemory::<f32>::new(128usize);let mut sv=SharedMemory::<f32>::new(128usize);let mut sn=SharedMemory::<u32>::new(128usize);
    let mut mu=0.0f32;let mut m2=0.0f32;let mut n=0u32;let mut j=lane*4;
    while j<width {
        #[unroll]
        for k in 0..4usize {
            let x=f32::cast_from(a[address(a,row*width+j+k)]);n+=1;let delta=x-mu;
            mu=fma(delta,1.0f32/f32::cast_from(n),mu);m2=fma(delta,x-mu,m2);
        }
        j+=512;
    }
    sm[lane]=mu;sv[lane]=m2;sn[lane]=n;sync_ruda();let mut stride=16usize;
    while stride>0 {
        if lane%32<stride {
            let (m,v,count)=vector_merge(sm[lane],sv[lane],sn[lane],sm[lane+stride],sv[lane+stride],sn[lane+stride]);
            sm[lane]=m;sv[lane]=v;sn[lane]=count;
        }
        sync_ruda();stride/=2;
    }
    let mut stride=2usize;
    while stride>0 {
        if lane%32==0 && lane/32<stride {
            let other=lane+stride*32;let (m,v,count)=vector_merge(sm[lane],sv[lane],sn[lane],sm[other],sv[other],sn[other]);
            sm[lane]=m;sv[lane]=v;sn[lane]=count;
        }
        sync_ruda();stride/=2;
    }
    if lane==0 {let v=sv[0]/f32::cast_from(width);mean[row]=sm[0];variance[row]=v;rstd[row]=cuda_inverse_sqrt::<f32>(v+eps);}
}

#[ruda(launch)]
fn statistics<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,mean:&mut Tensor<F>,variance:&mut Tensor<F>,rstd:&mut Tensor<F>,eps:F,#[comptime] kind:u32) {
    let row=RUDA_POS_X as usize;let lane=UNIT_POS as usize;let width=a.shape(1);
    let mut sm=SharedMemory::<F>::new(256usize);let mut sv=SharedMemory::<F>::new(256usize);let mut sn=SharedMemory::<u32>::new(256usize);
    let mut mu=F::new(0.0);let mut m2=F::new(0.0);let mut n=0u32;let mut j=lane;
    while j<width {
        let x=F::cast_from(a[address(a,row*width+j)]);n+=1;
        if comptime!(kind==3) {m2+=x*x;}
        else {let d=x-mu;mu+=d/F::cast_from(n);m2+=d*(x-mu);}
        j+=256;
    }
    sm[lane]=mu;sv[lane]=m2;sn[lane]=n;sync_ruda();let mut stride=128usize;
    while stride>0 {
        if lane<stride {
            let na=sn[lane];let nb=sn[lane+stride];
            if nb>0 {
                if na==0 {sm[lane]=sm[lane+stride];sv[lane]=sv[lane+stride];}
                else {
                    if comptime!(kind==3) {sv[lane]=sv[lane]+sv[lane+stride];}
                    else {let delta=sm[lane+stride]-sm[lane];let ratio=F::cast_from(nb)/F::cast_from(na+nb);sm[lane]=sm[lane]+delta*ratio;sv[lane]=sv[lane]+sv[lane+stride]+delta*delta*F::cast_from(na)*ratio;}
                }
                sn[lane]=na+nb;
            }
        }
        sync_ruda();stride/=2;
    }
    if lane==0 {
        let v=sv[0]/F::cast_from(width);let mu=sm[0];
        let mut epsilon=eps;if comptime!(kind==1) {epsilon=F::cast_from(S::cast_from(eps));}
        let mut inv=(v+epsilon).inverse_sqrt();
        if comptime!(kind!=4) {inv=cuda_inverse_sqrt::<F>(v+epsilon);}
        if comptime!(kind==4) {if v==F::new(0.0) && eps==F::new(0.0) {inv=F::new(0.0);}}
        mean[row]=mu;variance[row]=v;rstd[row]=inv;
    }
}

#[ruda]
fn affine_index(row:usize,col:usize,channels:usize,spatial:usize,groups:usize,#[comptime] kind:u32)->usize {
    let mut index=col;
    if comptime!(kind==1) {index=(row%groups)*(channels/groups)+col/spatial;}
    else if comptime!(kind==2) {index=row;}
    index
}

#[ruda(launch)]
fn affine<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,mean:&Tensor<F>,rstd:&Tensor<F>,weight:&Tensor<F>,bias:&Tensor<F>,out:&mut Tensor<S>,channels:usize,spatial:usize,groups:usize,#[comptime] kind:u32,#[comptime] weighted:bool,#[comptime] biased:bool) {
    let p=ABSOLUTE_POS as usize;
    if p<out.len() {
        let width=a.shape(1);let row=p/width;let col=p%width;let index=affine_index(row,col,channels,spatial,groups,kind);
        let x=F::cast_from(a[address(a,p)]);let mu=mean[row];let inv=rstd[row];let mut w=F::new(1.0);let mut b=F::new(0.0);
        if comptime!(weighted) {w=weight[index];}if comptime!(biased) {b=bias[index];}
        let mut y=(x-mu)*inv*w+b;
        if comptime!(kind==3) {y=x*inv*w;}
        else if comptime!(kind==2) {y=(x-mu)*w*inv+b;}
        else if comptime!(kind==1 && (weighted || biased)) {if spatial>1 {let scale=inv*w;y=fma(scale,x,fma(-scale,mu,b));}}
        out[address(out,p)]=S::cast_from(y);
    }
}

#[ruda(launch)]
fn input_gradient<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,grad:&Tensor<S>,mean:&Tensor<F>,rstd:&Tensor<F>,weight:&Tensor<F>,out:&mut Tensor<S>,channels:usize,spatial:usize,groups:usize,#[comptime] kind:u32,#[comptime] weighted:bool,#[comptime] training:bool) {
    let row=RUDA_POS_X as usize;let lane=UNIT_POS as usize;let width=a.shape(1);
    let mut sm=SharedMemory::<F>::new(256usize);let mut sd=SharedMemory::<F>::new(256usize);
    let mut mu=mean[row];if comptime!(kind==3) {mu=F::new(0.0);}let inv=rstd[row];
    let mut sum=F::new(0.0);let mut dot=F::new(0.0);let mut j=lane;
    if comptime!(training) {
        while j<width {
            let p=row*width+j;let dy=F::cast_from(grad[address(grad,p)]);let x=F::cast_from(a[address(a,p)]);let mut w=F::new(1.0);
            if comptime!(weighted) {w=weight[affine_index(row,j,channels,spatial,groups,kind)];}
            let g=dy*w;sum+=g;
            if comptime!(kind==1) {let mut product=dy*x;if spatial==1 {product=F::cast_from(S::cast_from(product));}dot+=product*w;}
            else {dot+=g*(x-mu);}
            j+=256;
        }
    }
    sm[lane]=sum;sd[lane]=dot;sync_ruda();let mut stride=128usize;
    while stride>0 {if lane<stride {sm[lane]=sm[lane]+sm[lane+stride];sd[lane]=sd[lane]+sd[lane+stride];}sync_ruda();stride/=2;}
    let average=sm[0]/F::cast_from(width);let correction=sd[0]*inv*inv/F::cast_from(width);let mut j=lane;
    while j<width {
        let p=row*width+j;let mut g=F::cast_from(grad[address(grad,p)]);
        if comptime!(weighted) {g*=weight[affine_index(row,j,channels,spatial,groups,kind)];}
        let mut result=g*inv;
        if comptime!(kind==1) {
            let scale=F::new(1.0)/F::cast_from(width);let c2=fma(sm[0],mu,-sd[0])*inv*inv*inv*scale;let c3=fma(-c2,mu,-(sm[0]*inv*scale));
            let mut c1=inv;if comptime!(weighted) {c1*=weight[affine_index(row,j,channels,spatial,groups,kind)];}
            result=fma(c1,F::cast_from(grad[address(grad,p)]),c2*F::cast_from(a[address(a,p)]))+c3;
        } else {
            if comptime!(training) {if comptime!(kind!=3) {g-=average;}g-=(F::cast_from(a[address(a,p)])-mu)*correction;}
            result=g*inv;
        }
        out[address(out,p)]=S::cast_from(result);j+=256;
    }
}

#[ruda(launch)]
fn affine_gradient<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,grad:&Tensor<S>,mean:&Tensor<F>,rstd:&Tensor<F>,dw:&mut Tensor<F>,db:&mut Tensor<F>,channels:usize,spatial:usize,groups:usize,#[comptime] kind:u32) {
    let index=RUDA_POS_X as usize;let lane=UNIT_POS as usize;let width=a.shape(1);let rows=a.shape(0);
    let mut sw=SharedMemory::<F>::new(256usize);let mut sb=SharedMemory::<F>::new(256usize);
    let mut count=rows;if comptime!(kind==1) {count=rows/groups*spatial;}else if comptime!(kind==2) {count=width;}
    let mut w=F::new(0.0);let mut b=F::new(0.0);let mut j=lane;
    while j<count {
        let mut row=j;let mut col=index;
        if comptime!(kind==1) {row=j/spatial*groups+index/(channels/groups);col=index%(channels/groups)*spatial+j%spatial;}
        else if comptime!(kind==2) {row=index;col=j;}
        let p=row*width+col;let g=F::cast_from(grad[address(grad,p)]);let mut x=F::cast_from(a[address(a,p)]);
        if comptime!(kind!=3) {x-=mean[row];}
        w+=g*x*rstd[row];b+=g;j+=256;
    }
    sw[lane]=w;sb[lane]=b;sync_ruda();let mut stride=128usize;
    while stride>0 {if lane<stride {sw[lane]=sw[lane]+sw[lane+stride];sb[lane]=sb[lane]+sb[lane+stride];}sync_ruda();stride/=2;}
    if lane==0 {dw[index]=sw[0];db[index]=sb[0];}
}

#[ruda]
fn batch_sums<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,grad:&Tensor<S>,mu:F,spatial:usize,bx:usize,by:usize)->(F,F) {
    let row=RUDA_POS_X as usize;let lane=UNIT_POS as usize;let width=a.shape(1);let threads=bx*by;
    let mut sm=SharedMemory::<F>::new(512usize);let mut sd=SharedMemory::<F>::new(512usize);
    let mut sum=F::new(0.0);let mut dot=F::new(0.0);let mut batch=lane/bx;
    while batch<width/spatial {
        let mut j=lane%bx;
        while j<spatial {
            let p=row*width+batch*spatial+j;let g=F::cast_from(grad[address(grad,p)]);let centered=F::cast_from(a[address(a,p)])-mu;
            sum+=g;dot+=F::cast_from(S::cast_from(g*centered));j+=bx;
        }
        batch+=by;
    }
    sm[lane]=sum;sd[lane]=dot;sync_ruda();let mut stride=16usize;
    while stride>0 {
        if lane%32<stride {sm[lane]=sm[lane]+F::cast_from(S::cast_from(sm[lane+stride]));sd[lane]=sd[lane]+F::cast_from(S::cast_from(sd[lane+stride]));}
        sync_ruda();stride/=2;
    }
    let mut sum=F::new(0.0);let mut dot=F::new(0.0);
    if lane<threads/32 {sum=sm[lane*32];dot=sd[lane*32];}
    sync_ruda();sm[lane]=sum;sd[lane]=dot;sync_ruda();let mut stride=16usize;
    while stride>0 {
        if lane<stride {sm[lane]=sm[lane]+F::cast_from(S::cast_from(sm[lane+stride]));sd[lane]=sd[lane]+F::cast_from(S::cast_from(sd[lane+stride]));}
        sync_ruda();stride/=2;
    }
    (sm[0],sd[0])
}

#[ruda(launch)]
fn batch_gradient<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,grad:&Tensor<S>,mean:&Tensor<F>,rstd:&Tensor<F>,weight:&Tensor<F>,dx:&mut Tensor<S>,dw:&mut Tensor<F>,db:&mut Tensor<F>,spatial:usize,bx:usize,by:usize,#[comptime] training:bool,#[comptime] weighted:bool,#[comptime] input_grad:bool) {
    let row=RUDA_POS_X as usize;let lane=UNIT_POS as usize;let width=a.shape(1);let threads=bx*by;let mu=mean[row];let inv=rstd[row];
    let (sum,dot)=batch_sums::<S,F>(a,grad,mu,spatial,bx,by);
    if lane==0 {dw[row]=dot*inv;db[row]=sum;}
    if comptime!(input_grad) {
        let norm=F::new(1.0)/F::cast_from(width);let average=sum*norm;let projection=dot*norm*inv*inv;let mut scale=inv;
        if comptime!(weighted) {scale*=weight[row];}
        let mut j=lane;
        while j<width {
            let p=row*width+j;let mut g=F::cast_from(grad[address(grad,p)]);
            if comptime!(training) {g=g-(F::cast_from(a[address(a,p)])-mu)*projection-average;}
            dx[address(dx,p)]=S::cast_from(g*scale);j+=threads;
        }
    }
}

#[ruda(launch)]
fn batch_gather<S:Float+RudaElement,F:Float+RudaElement>(means:&Tensor<F>,invstds:&Tensor<F>,counts:&Tensor<S>,running_mean:&mut Tensor<S>,running_var:&mut Tensor<S>,mean:&mut Tensor<F>,invstd:&mut Tensor<F>,eps:F,momentum:F,#[comptime] wide_count:bool) {
    let channel=ABSOLUTE_POS as usize;let channels=means.shape(1);
    if channel<channels {
        let mut avg=F::new(0.0);let mut m2=F::new(0.0);let mut n=0i64;
        for rank in 0..means.shape(0) {
            let count=F::cast_from(counts[address(counts,rank)]);let p=rank*channels+channel;
            let m=means[address(means,p)];let sigma=F::new(1.0)/invstds[address(invstds,p)];
            let nf=F::cast_from(n);let factor=F::new(1.0)/(nf+count);let delta=avg-m;
            m2+=(sigma*sigma-eps)*count+delta*delta*nf*count*factor;
            avg=nf*factor*avg+count*factor*m;
            if comptime!(wide_count) {n=i64::cast_from(nf+count);}
            else {n=i64::cast_from(i32::cast_from(nf+count));}
        }
        mean[channel]=avg;invstd[channel]=F::new(1.0)/(m2/F::cast_from(n)+eps).sqrt();
        if running_mean.len()>0 {
            let p=address(running_mean,channel);running_mean[p]=S::cast_from((F::new(1.0)-momentum)*F::cast_from(running_mean[p])+momentum*avg);
        }
        if running_var.len()>0 {
            let p=address(running_var,channel);running_var[p]=S::cast_from((F::new(1.0)-momentum)*F::cast_from(running_var[p])+momentum*(m2/F::cast_from(n-1)));
        }
    }
}

#[ruda(launch)]
fn batch_reduce<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,grad:&Tensor<S>,mean:&Tensor<F>,invstd:&Tensor<F>,sum_dy:&mut Tensor<F>,sum_dy_xmu:&mut Tensor<F>,dw:&mut Tensor<F>,db:&mut Tensor<F>,spatial:usize,bx:usize,by:usize) {
    let row=RUDA_POS_X as usize;let lane=UNIT_POS as usize;
    let (sum,dot)=batch_sums::<S,F>(a,grad,mean[row],spatial,bx,by);
    if lane==0 {
        if sum_dy.len()>0 {sum_dy[row]=sum;sum_dy_xmu[row]=dot;}
        if dw.len()>0 {dw[row]=dot*invstd[row];}
        if db.len()>0 {db[row]=sum;}
    }
}

#[ruda(launch)]
fn batch_reduce_channels_last<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,grad:&Tensor<S>,mean:&Tensor<F>,staging:&mut Tensor<F>,bx:usize,by:usize,parts:usize) {
    let task=RUDA_POS_X as usize;let lane=UNIT_POS as usize;let x=lane%bx;let y=lane/bx;
    let channel=task/parts*bx+x;let part=task%parts;let channels=a.shape(0);let width=a.shape(1);
    let step=by*parts;let mut j=part*by+y;let mut mu=F::new(0.0);
    if channel<channels {mu=mean[channel];}
    let mut sums=Array::<F>::new(4usize);let mut dots=Array::<F>::new(4usize);
    #[unroll]
    for k in 0..4usize {sums[k]=F::new(0.0);dots[k]=F::new(0.0);}
    let loops=(width+step*4-1)/(step*4);
    for _iteration in 0..loops {
        #[unroll]
        for k in 0..4usize {
            let mut value=F::new(0.0);let mut dy=F::new(0.0);
            if channel<channels && j<width {let p=channel*width+j;value=F::cast_from(a[address(a,p)]);dy=F::cast_from(grad[address(grad,p)]);}
            sums[k]=sums[k]+dy;dots[k]=dots[k]+dy*(value-mu);j+=step;
        }
    }
    let mut sum=sums[0];let mut dot=dots[0];
    #[unroll]
    for k in 1..4usize {sum+=sums[k];dot+=dots[k];}
    let mut sm=SharedMemory::<F>::new(512usize);let mut sd=SharedMemory::<F>::new(512usize);
    sm[lane]=sum;sd[lane]=dot;sync_ruda();let mut stride=by/2;
    while stride>0 {
        if y<stride {sm[lane]=sm[lane]+sm[lane+stride*bx];sd[lane]=sd[lane]+sd[lane+stride*bx];}
        sync_ruda();stride/=2;
    }
    if y==0 && channel<channels {staging[part*channels+channel]=sm[lane];staging[(parts+part)*channels+channel]=sd[lane];}
}

#[ruda(launch)]
fn batch_reduce_channels_last_finish<F:Float+RudaElement>(staging:&Tensor<F>,invstd:&Tensor<F>,sum_dy:&mut Tensor<F>,sum_dy_xmu:&mut Tensor<F>,dw:&mut Tensor<F>,db:&mut Tensor<F>,bx:usize,by:usize,parts:usize) {
    let lane=UNIT_POS as usize;let x=lane%bx;let y=lane/bx;let channel=RUDA_POS_X as usize*bx+x;let channels=invstd.len();
    let mut sum=F::new(0.0);let mut dot=F::new(0.0);let mut part=y;
    while part<parts {
        if channel<channels {sum+=staging[part*channels+channel];dot+=staging[(parts+part)*channels+channel];}
        part+=by;
    }
    let mut sm=SharedMemory::<F>::new(512usize);let mut sd=SharedMemory::<F>::new(512usize);
    sm[lane]=sum;sd[lane]=dot;sync_ruda();let mut stride=by/2;
    while stride>0 {
        if y<stride {sm[lane]=sm[lane]+sm[lane+stride*bx];sd[lane]=sd[lane]+sd[lane+stride*bx];}
        sync_ruda();stride/=2;
    }
    if y==0 && channel<channels {
        let mut sum=sm[lane];let mut dot=sd[lane];
        if parts==1 {sum=staging[channel];dot=staging[channels+channel];}
        if sum_dy.len()>0 {sum_dy[channel]=sum;sum_dy_xmu[channel]=dot;}
        if dw.len()>0 {dw[channel]=dot*invstd[channel];}
        if db.len()>0 {db[channel]=sum;}
    }
}

#[ruda(launch)]
fn batch_backward_element<S:Float+RudaElement,F:Float+RudaElement>(a:&Tensor<S>,grad:&Tensor<S>,mean:&Tensor<F>,invstd:&Tensor<F>,weight:&Tensor<F>,sum_dy:&Tensor<F>,sum_dy_xmu:&Tensor<F>,counts:&Tensor<i32>,dx:&mut Tensor<S>,#[comptime] weighted:bool) {
    let p=ABSOLUTE_POS as usize;let lane=UNIT_POS as usize;let mut total=SharedMemory::<i64>::new(1usize);
    if lane==0 {let mut n=0i64;for j in 0..counts.len() {n+=i64::cast_from(counts[address(counts,j)]);}total[0]=n;}
    sync_ruda();
    if p<dx.len() {
        let row=p/a.shape(1);let norm=F::new(1.0)/F::cast_from(total[0]);let inv=invstd[row];let mut scale=inv;
        if comptime!(weighted) {scale*=weight[row];}
        let average=sum_dy[row]*norm;let projection=inv*inv*sum_dy_xmu[row]*norm;
        let g=F::cast_from(grad[address(grad,p)]);let centered=F::cast_from(a[address(a,p)])-mean[row];
        dx[address(dx,p)]=S::cast_from((g-average-centered*projection)*scale);
    }
}

#[ruda(launch)]
fn running_update<S:Float+RudaElement,F:Float+RudaElement>(mean:&Tensor<F>,variance:&Tensor<F>,running_mean:&mut Tensor<S>,running_var:&mut Tensor<S>,momentum:F,correction:F) {
    let p=ABSOLUTE_POS as usize;
    if p<mean.len() {
        let old_mean=F::cast_from(running_mean[address(running_mean,p)]);let old_var=F::cast_from(running_var[address(running_var,p)]);
        running_mean[address(running_mean,p)]=S::cast_from(mean[p]*momentum+(F::new(1.0)-momentum)*old_mean);
        running_var[address(running_var,p)]=S::cast_from(variance[p]*correction*momentum+(F::new(1.0)-momentum)*old_var);
    }
}

fn dispatch_typed<S:Float+RudaElement,F:Float+RudaElement>(c:&ComputeClient<CudaRuntime>,op:u32,xs:&[View],ints:&[i64],values:&[f64],epsilon:F,momentum:F,correction:F) {
    let a=&xs[0];let rows=a.shape[0];let block=RudaDim::new_1d(256);
    let grid=|n:usize|RudaCount::Static(u32::try_from(n).expect("normalization grid overflow"),1,1);
    if op==315 {
        if xs[5].len>0 {unsafe {batch_gather::launch::<S,F,CudaRuntime>(c,grid(xs[5].len.div_ceil(256)),block,a.arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),xs[6].arg(),epsilon,correction,ints[0]!=0);}}
        return;
    }
    if op==316 {
        let bx=ints[1] as usize;let by=ints[2] as usize;let parts=ints[3] as usize;let block=RudaDim::new_1d((bx*by) as u32);
        if rows>0 {unsafe {
            if ints[4]!=0 {
                batch_reduce_channels_last::launch::<S,F,CudaRuntime>(c,grid(rows.div_ceil(bx)*parts),block,a.arg(),xs[1].arg(),xs[2].arg(),xs[8].arg(),bx,by,parts);
                batch_reduce_channels_last_finish::launch::<F,CudaRuntime>(c,grid(rows.div_ceil(bx)),block,xs[8].arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),xs[6].arg(),xs[7].arg(),bx,by,parts);
            } else {
                batch_reduce::launch::<S,F,CudaRuntime>(c,grid(rows),block,a.arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),xs[6].arg(),xs[7].arg(),ints[0] as usize,bx,by);
            }
        }}
        return;
    }
    if op==317 {
        if a.len>0 {unsafe {batch_backward_element::launch::<S,F,CudaRuntime>(c,grid(a.len.div_ceil(256)),block,a.arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),xs[6].arg(),xs[7].arg(),xs[8].arg(),ints[0]!=0);}}
        return;
    }
    if op==314 {
        let bx=ints[1] as usize;let by=ints[2] as usize;
        if a.len>0 {unsafe {batch_gradient::launch::<S,F,CudaRuntime>(c,grid(rows),RudaDim::new_1d((bx*by) as u32),a.arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),xs[6].arg(),xs[7].arg(),ints[0] as usize,bx,by,ints[3]!=0,ints[4]!=0,ints[5]!=0);}}
        return;
    }
    if op==313 {
        if a.len>0 {unsafe {running_update::launch::<S,F,CudaRuntime>(c,grid(a.len.div_ceil(256)),block,a.arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),momentum,correction);}}
        return;
    }
    let kind=ints[0] as u32;
    if op==309 {
        if rows>0 {unsafe {
            if ints.len()==2 && ints[1]!=0 {vector_statistics::launch::<S,CudaRuntime>(c,grid(rows),RudaDim::new_1d(128),a.arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),values[0] as f32);}
            else {statistics::launch::<S,F,CudaRuntime>(c,grid(rows),block,a.arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),epsilon,kind);}
        }}
    } else {
        let channels=ints[1] as usize;let spatial=ints[2] as usize;let groups=ints[3] as usize;
        if op==310 {
            if a.len>0 {unsafe {affine::launch::<S,F,CudaRuntime>(c,grid(a.len.div_ceil(256)),block,a.arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),channels,spatial,groups,kind,ints[4]!=0,ints[5]!=0);}}
        } else if op==311 {
            if a.len>0 {unsafe {input_gradient::launch::<S,F,CudaRuntime>(c,grid(rows),block,a.arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),channels,spatial,groups,kind,ints[4]!=0,ints[5]!=0);}}
        } else {
            if xs[4].len>0 {unsafe {affine_gradient::launch::<S,F,CudaRuntime>(c,grid(xs[4].len),block,a.arg(),xs[1].arg(),xs[2].arg(),xs[3].arg(),xs[4].arg(),xs[5].arg(),channels,spatial,groups,kind);}}
        }
    }
    let _=values;
}

pub(super) fn dispatch(op:u32,xs:&[View],ints:&[i64],values:&[f64]) {
    assert_eq!(xs.len(),match op {315=>7,316|317=>9,314=>8,309|313=>4,_=>6});
    let dtype=if op==313 || op==315 {xs[2].dtype}else{xs[0].dtype};let acc=if dtype==9 {9}else{0};
    if op==315 {
        assert_eq!(ints.len(),1);assert_eq!(values.len(),2);assert_eq!(xs[0].shape.len(),2);assert_eq!(xs[0].shape,xs[1].shape);
        let ranks=xs[0].shape[0];let channels=xs[0].shape[1];
        for i in [0,1,5,6] {assert_eq!(xs[i].dtype,acc);}
        assert_eq!(xs[2].shape,vec![ranks]);
        for i in [3,4] {assert_eq!(xs[i].dtype,dtype);assert!(xs[i].len==0 || xs[i].shape==vec![channels]);}
        for i in [5,6] {assert_eq!(xs[i].shape,vec![channels]);}
    } else if op==316 || op==317 {
        assert_eq!(xs[0].shape.len(),2);assert_eq!(xs[0].shape,xs[1].shape);assert_eq!(xs[1].dtype,dtype);assert!(values.is_empty());
        let channels=xs[0].shape[0];
        for i in [2,3] {assert_eq!(xs[i].dtype,acc);assert_eq!(xs[i].len,channels);}
        if op==316 {
            assert_eq!(ints.len(),5);assert!(ints[0]>0 && ints[1]>0 && ints[2]>0 && ints[1]*ints[2]<=512 && ints[3]>0);
            assert!((ints[1] as u64).is_power_of_two() && (ints[2] as u64).is_power_of_two());
            assert_eq!(xs[0].shape[1]%(ints[0] as usize),0);
            for i in 4..8 {assert_eq!(xs[i].dtype,acc);assert!(xs[i].len==0 || xs[i].len==channels);}
            assert_eq!(xs[4].len,xs[5].len);assert_eq!(xs[8].dtype,acc);
            if ints[4]!=0 {assert_eq!(xs[8].len,2*channels*ints[3] as usize);}
            else {assert!(ints[1]>=32);assert_eq!(ints[3],1);}
        } else {
            assert_eq!(ints.len(),1);assert_eq!(xs[4].dtype,acc);if ints[0]!=0 {assert_eq!(xs[4].len,channels);}
            for i in [5,6] {assert_eq!(xs[i].dtype,acc);assert_eq!(xs[i].len,channels);}
            assert_eq!(xs[7].dtype,5);assert_eq!(xs[8].dtype,dtype);assert_eq!(xs[8].shape,xs[0].shape);
        }
    } else if op==313 {assert_eq!(values.len(),2);assert_eq!(xs[0].dtype,acc);assert_eq!(xs[1].dtype,acc);assert_eq!(xs[2].dtype,xs[3].dtype);for x in &xs[1..] {assert_eq!(x.len,xs[0].len);}}
    else if op==314 {
        assert_eq!(xs[0].shape.len(),2);assert_eq!(xs[0].shape,xs[1].shape);assert_eq!(xs[1].dtype,dtype);assert_eq!(ints.len(),6);
        assert!(ints[0]>0 && ints[1]>=32 && ints[2]>0 && ints[1]*ints[2]<=512);assert_eq!(ints[1]%32,0);assert_eq!(xs[0].shape[1]%(ints[0] as usize),0);
        for i in [2,3,6,7] {assert_eq!(xs[i].dtype,acc);assert_eq!(xs[i].len,xs[0].shape[0]);}
        assert_eq!(xs[4].dtype,acc);if ints[4]!=0 {assert_eq!(xs[4].len,xs[0].shape[0]);}
        if ints[5]!=0 {assert_eq!(xs[5].dtype,dtype);assert_eq!(xs[5].shape,xs[0].shape);}
    } else {
        assert_eq!(xs[0].shape.len(),2);assert!(ints.len()>=1);assert!((0..=4).contains(&ints[0]));
        let rows=xs[0].shape[0];
        if op==309 {assert_eq!(values.len(),1);if ints.len()==2 && ints[1]!=0 {assert_eq!(ints[0],0);assert_eq!(acc,0);assert_eq!(xs[0].shape[1]%4,0);}for x in &xs[1..] {assert_eq!(x.dtype,acc);assert_eq!(x.len,rows);}}
        else {
            assert_eq!(ints.len(),6);assert!(ints[1]>=0 && ints[2]>=0 && ints[3]>0);
            let stats=if op==310 {1}else{2};
            for x in &xs[stats..stats+2] {assert_eq!(x.dtype,acc);assert_eq!(x.len,rows);}
            if op!=310 {assert_eq!(xs[1].dtype,dtype);assert_eq!(xs[1].shape,xs[0].shape);}
            if op==312 {assert_eq!(xs[4].dtype,acc);assert_eq!(xs[5].dtype,acc);assert_eq!(xs[4].len,xs[5].len);}
            else {assert_eq!(xs[5].dtype,dtype);assert_eq!(xs[5].shape,xs[0].shape);for x in &xs[stats+2..5] {assert_eq!(x.dtype,acc);}}
        }
    }
    let c=client();let epsilon=values.first().copied().unwrap_or(0.0);let correction=values.get(1).copied().unwrap_or(0.0);
    macro_rules! run {($s:ty,$f:ty)=>{dispatch_typed::<$s,$f>(&c,op,xs,ints,values,epsilon as $f,epsilon as $f,correction as $f)};}
    match dtype {0=>run!(f32,f32),1=>run!(f16,f32),2=>run!(bf16,f32),9=>run!(f64,f64),_=>panic!("normalization requires real floating storage")}
    LAUNCHES.fetch_add(1,Ordering::Relaxed);finish_dispatch(&c);
}
