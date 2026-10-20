use core::marker::PhantomData;

use ruda_tensor::{Backend, tensor::FloatTensor};

use crate::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};

/// A value-preserving transfer and its reverse gradient transfer between native backends.
pub trait DifferentiableTransfer<Src: Backend, Dst: Backend>:
    Send + core::fmt::Debug + 'static
{
    /// Transfer the forward value, preserving its shape and dtype.
    fn forward(tensor: FloatTensor<Src>, device: &Dst::Device) -> FloatTensor<Dst>;
    /// Transfer the incoming gradient back to the original device.
    fn backward(tensor: FloatTensor<Dst>, device: &Src::Device) -> FloatTensor<Src>;
}

#[derive(Debug)]
struct Transfer<Src, Dst, Adapter>(PhantomData<(Src, Dst, Adapter)>);

impl<Src: Backend, Dst: Backend, Adapter: DifferentiableTransfer<Src, Dst>> Backward<Dst, 1>
    for Transfer<Src, Dst, Adapter>
{
    type State = Src::Device;

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        let gradient = grads.consume::<Dst>(&ops.node);
        if let Some(parent) = &ops.parents[0] {
            grads.register::<Src>(parent.id, Adapter::backward(gradient, &ops.state));
        }
    }
}

impl<Src: Backend, C: CheckpointStrategy> Autodiff<Src, C> {
    /// Move a tensor between native backends without severing its gradient graph.
    /// Transfers retain their result rather than replaying a device copy during checkpointing.
    pub fn to_backend<Dst: Backend, Adapter: DifferentiableTransfer<Src, Dst>>(
        tensor: FloatTensor<Self>,
        device: &Dst::Device,
    ) -> FloatTensor<Autodiff<Dst, C>> {
        let source_device = Src::float_device(&tensor.primitive);
        let context = tensor.context::<C>();
        let prep = Transfer::<Src, Dst, Adapter>(PhantomData)
            .prepare::<C>([tensor.node.clone()])
            .compute_bound()
            .stateful();
        let output = Adapter::forward(tensor.primitive, device);
        let output = match prep {
            OpsKind::Tracked(prep) => prep.finish(source_device, output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        };
        output.with_context::<C>(context)
    }

    /// Transfer quantized storage and its optional differentiable surrogate independently.
    pub fn quantized_to_backend<Dst: Backend, Adapter: DifferentiableTransfer<Src, Dst>>(
        tensor: ruda_tensor::tensor::QuantizedTensor<Self>,
        device: &Dst::Device,
        transfer_storage: impl FnOnce(Src::QuantizedTensorPrimitive, &Dst::Device) -> Dst::QuantizedTensorPrimitive,
    ) -> ruda_tensor::tensor::QuantizedTensor<Autodiff<Dst, C>> {
        let primitive = transfer_storage(tensor.primitive, device);
        let surrogate = tensor.surrogate.map(|value| Self::to_backend::<Dst, Adapter>(value, device));
        crate::tensor::AutodiffQTensor {
            primitive,
            surrogate,
            checkpointing: tensor.checkpointing,
            autodiff_enabled: tensor.autodiff_enabled,
        }
    }
}
