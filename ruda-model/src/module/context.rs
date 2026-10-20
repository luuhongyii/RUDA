use super::{Module, ModuleMapper, Param, Parameter};
use ruda_tensor::{AutodiffContext, Backend, Bool, GradientCheckpointingStrategy, Int, Tensor};

#[doc(hidden)]
pub fn has_autodiff<B: Backend, M: Module<B>>(module: &M) -> bool {
    struct Inspector(bool);
    impl Inspector {
        fn parameter<T: Parameter>(&mut self, param: Param<T>) -> Param<T> {
            self.0 |= param.lazy_autodiff_context().is_some_and(|context| context != AutodiffContext::Disabled);
            param
        }
    }
    impl<B: Backend> ModuleMapper<B> for Inspector {
        fn map_tensor<const D: usize, K: ruda_tensor::BasicOps<B>>(&mut self, tensor: Tensor<B, D, K>) -> Tensor<B, D, K> {
            let primitive = tensor.into_primitive();
            self.0 |= K::autodiff_context(&primitive) != AutodiffContext::Disabled;
            Tensor::from_primitive(primitive)
        }
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> { self.parameter(param) }
        fn map_int<const D: usize>(&mut self, param: Param<Tensor<B, D, Int>>) -> Param<Tensor<B, D, Int>> { self.parameter(param) }
        fn map_bool<const D: usize>(&mut self, param: Param<Tensor<B, D, Bool>>) -> Param<Tensor<B, D, Bool>> { self.parameter(param) }
    }
    let mut inspector = Inspector(false);
    let _ = module.clone().map(&mut inspector);
    inspector.0
}

pub(super) enum ContextMapper {
    Association(AutodiffContext),
    Checkpointing(GradientCheckpointingStrategy),
}

impl<B: Backend> ModuleMapper<B> for ContextMapper {
    fn map_tensor<const D: usize, K: ruda_tensor::BasicOps<B>>(&mut self, tensor: Tensor<B, D, K>) -> Tensor<B, D, K> {
        let primitive = tensor.into_primitive();
        let context = match *self {
            Self::Association(context) => context,
            Self::Checkpointing(policy) => {
                assert!(matches!(K::autodiff_context(&primitive), AutodiffContext::Enabled(_)),
                    "Checkpointing requires an autodiff-associated tensor");
                AutodiffContext::Enabled(policy)
            }
        };
        Tensor::from_primitive(K::with_autodiff_context(primitive, context))
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let planned = param.require_grad;
        let mut mapped = param.map(|tensor| match *self {
            Self::Association(context) => {
                let tensor = tensor.with_autodiff_context(context);
                if tensor.is_autodiff() { tensor.set_require_grad(planned) } else { tensor }
            }
            Self::Checkpointing(policy) => tensor.with_gradient_checkpointing_strategy(policy),
        });
        mapped.require_grad = planned;
        mapped
    }

    fn map_int<const D: usize>(&mut self, param: Param<Tensor<B, D, Int>>) -> Param<Tensor<B, D, Int>> {
        param.map(|tensor| match *self {
            Self::Association(context) => tensor.with_autodiff_context(context),
            Self::Checkpointing(policy) => tensor.with_gradient_checkpointing_strategy(policy),
        })
    }

    fn map_bool<const D: usize>(&mut self, param: Param<Tensor<B, D, Bool>>) -> Param<Tensor<B, D, Bool>> {
        param.map(|tensor| match *self {
            Self::Association(context) => tensor.with_autodiff_context(context),
            Self::Checkpointing(policy) => tensor.with_gradient_checkpointing_strategy(policy),
        })
    }
}
