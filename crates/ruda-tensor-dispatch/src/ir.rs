use ruda_tensor::graph::{BackendIr, HandleKind, TensorHandle};
use crate::{Dispatch, DispatchTensor};

impl BackendIr for Dispatch {
    type Handle = HandleKind<Self>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> DispatchTensor {
        match handle.handle {
            HandleKind::Float(tensor) => tensor,
            other => panic!("Expected float handle, got {}", other.name()),
        }
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> DispatchTensor {
        match handle.handle {
            HandleKind::Int(tensor) => tensor,
            other => panic!("Expected integer handle, got {}", other.name()),
        }
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> DispatchTensor {
        match handle.handle {
            HandleKind::Bool(tensor) => tensor,
            other => panic!("Expected boolean handle, got {}", other.name()),
        }
    }

    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> DispatchTensor {
        match handle.handle {
            HandleKind::Quantized(tensor) => tensor,
            other => panic!("Expected quantized handle, got {}", other.name()),
        }
    }

    fn float_tensor_handle(tensor: DispatchTensor) -> Self::Handle { HandleKind::Float(tensor) }
    fn int_tensor_handle(tensor: DispatchTensor) -> Self::Handle { HandleKind::Int(tensor) }
    fn bool_tensor_handle(tensor: DispatchTensor) -> Self::Handle { HandleKind::Bool(tensor) }
    fn quantized_tensor_handle(tensor: DispatchTensor) -> Self::Handle { HandleKind::Quantized(tensor) }
}
