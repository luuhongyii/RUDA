
use super::GradientsParams;
use ruda_model::module::{AutodiffModule, ModuleVisitor, Param, ParamId};
use ruda_model::tensor::{Tensor, backend::AutodiffBackend};
use ruda_model::tensor::ops::FloatTensorOps;
use core::marker::PhantomData;
use hashbrown::HashSet;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

pub struct GradientsParamsConverter<'a, M: AutodiffModule<B>, B: AutodiffBackend> {
    grads: &'a mut B::Gradients,
    grads_params: &'a mut GradientsParams,
    phatom: PhantomData<M>,
    filter: Option<HashSet<ParamId>>,
}

impl<'a, M: AutodiffModule<B>, B: AutodiffBackend> GradientsParamsConverter<'a, M, B> {
    /// Create a gradient visitor with an optional parameter-ID selection.
    pub fn new(
        grads: &'a mut B::Gradients,
        grads_params: &'a mut GradientsParams,
        filter: Option<Vec<ParamId>>,
    ) -> Self {
        Self {
            grads,
            grads_params,
            phatom: PhantomData,
            filter: filter.map(|ids| ids.into_iter().collect()),
        }
    }
}

#[derive(new)]
pub struct GradientsParamsChangeDevice<'a, M: AutodiffModule<B>, B: AutodiffBackend> {
    device: &'a B::Device,
    grads: &'a mut GradientsParams,
    phatom: PhantomData<M>,
}

impl<B, M> ModuleVisitor<B> for GradientsParamsConverter<'_, M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        use ruda_model::tensor::TensorPrimitive;
        let mut extract = |id: ParamId, value: &dyn core::any::Any| {
            if self.filter.as_ref().is_some_and(|filter| !filter.contains(&id)) { return; }
            let value = value.downcast_ref::<TensorPrimitive<B>>()
                .expect("Reparameterization parameter backend mismatch");
            let grad = match value {
                TensorPrimitive::Float(value) => {
                    if !B::float_is_require_grad(value) { return; }
                    B::grad_remove(value, self.grads)
                }
                TensorPrimitive::QFloat(value) => B::q_grad_remove(value, self.grads),
            };
            let Some(grad) = grad else { return; };
            let grad = match self.grads_params.container.remove::<B::InnerBackend>(&id) {
                Some(TensorPrimitive::Float(previous)) => B::InnerBackend::float_add(previous, grad),
                Some(TensorPrimitive::QFloat(_)) => panic!("Parameter gradient cannot be quantized"),
                None => grad,
            };
            self.grads_params.container.register::<B::InnerBackend>(id, TensorPrimitive::Float(grad));
        };
        extract(param.id, &param.base().into_primitive());
        param.visit_reparameterization_primitives(&mut extract);
    }
}

impl<B, M> ModuleVisitor<B> for GradientsParamsChangeDevice<'_, M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let Some(grad) = self.grads.remove::<B::InnerBackend, D>(param.id) else {
            return;
        };

        self.grads
            .register::<B::InnerBackend, D>(param.id, grad.to_device(self.device));
    }
}
