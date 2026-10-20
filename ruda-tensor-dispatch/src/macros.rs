macro_rules! dispatch {
    ($selection:expr, $call:ident) => {{
        let selection = $selection;
        match &selection.device {
crate::DispatchDevice::Host(device) => match selection.context {
    ruda_tensor::AutodiffContext::Disabled =>
        $call!(crate::backends::Host, Host, native, device, selection.context),
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Disabled) => {
        type Ad = ruda_autodiff::Autodiff<crate::backends::Host>;
        <Ad as ruda_tensor::Backend>::with_autodiff_creation_context(device, selection.context,
            || $call!(Ad, Host, ad, device, selection.context))
    }
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Balanced) => {
        type Ad = ruda_autodiff::Autodiff<crate::backends::Host, ruda_autodiff::checkpoint::strategy::BalancedCheckpointing>;
        <Ad as ruda_tensor::Backend>::with_autodiff_creation_context(device, selection.context,
            || $call!(Ad, Host, ad, device, selection.context))
    }
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Custom(_)) =>
        panic!("Dispatch accepts Disabled or Balanced checkpointing"),
},
#[cfg(feature = "cuda")]
crate::DispatchDevice::Cuda(device) => match selection.context {
    ruda_tensor::AutodiffContext::Disabled =>
        $call!(crate::backends::Cuda, Cuda, native, device, selection.context),
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Disabled) => {
        type Ad = ruda_autodiff::Autodiff<crate::backends::Cuda>;
        <Ad as ruda_tensor::Backend>::with_autodiff_creation_context(device, selection.context,
            || $call!(Ad, Cuda, ad, device, selection.context))
    }
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Balanced) => {
        type Ad = ruda_autodiff::Autodiff<crate::backends::Cuda, ruda_autodiff::checkpoint::strategy::BalancedCheckpointing>;
        <Ad as ruda_tensor::Backend>::with_autodiff_creation_context(device, selection.context,
            || $call!(Ad, Cuda, ad, device, selection.context))
    }
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Custom(_)) =>
        panic!("Dispatch accepts Disabled or Balanced checkpointing"),
},
#[cfg(feature = "cpu")]
crate::DispatchDevice::Cpu(device) => match selection.context {
    ruda_tensor::AutodiffContext::Disabled =>
        $call!(crate::backends::Cpu, Cpu, native, device, selection.context),
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Disabled) => {
        type Ad = ruda_autodiff::Autodiff<crate::backends::Cpu>;
        <Ad as ruda_tensor::Backend>::with_autodiff_creation_context(device, selection.context,
            || $call!(Ad, Cpu, ad, device, selection.context))
    }
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Balanced) => {
        type Ad = ruda_autodiff::Autodiff<crate::backends::Cpu, ruda_autodiff::checkpoint::strategy::BalancedCheckpointing>;
        <Ad as ruda_tensor::Backend>::with_autodiff_creation_context(device, selection.context,
            || $call!(Ad, Cpu, ad, device, selection.context))
    }
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Custom(_)) =>
        panic!("Dispatch accepts Disabled or Balanced checkpointing"),
},
#[cfg(feature = "wgpu")]
crate::DispatchDevice::Wgpu(device) => match selection.context {
    ruda_tensor::AutodiffContext::Disabled =>
        $call!(crate::backends::Wgpu, Wgpu, native, device, selection.context),
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Disabled) => {
        type Ad = ruda_autodiff::Autodiff<crate::backends::Wgpu>;
        <Ad as ruda_tensor::Backend>::with_autodiff_creation_context(device, selection.context,
            || $call!(Ad, Wgpu, ad, device, selection.context))
    }
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Balanced) => {
        type Ad = ruda_autodiff::Autodiff<crate::backends::Wgpu, ruda_autodiff::checkpoint::strategy::BalancedCheckpointing>;
        <Ad as ruda_tensor::Backend>::with_autodiff_creation_context(device, selection.context,
            || $call!(Ad, Wgpu, ad, device, selection.context))
    }
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Custom(_)) =>
        panic!("Dispatch accepts Disabled or Balanced checkpointing"),
},
#[cfg(feature = "rocm")]
crate::DispatchDevice::Rocm(device) => match selection.context {
    ruda_tensor::AutodiffContext::Disabled =>
        $call!(crate::backends::Rocm, Rocm, native, device, selection.context),
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Disabled) => {
        type Ad = ruda_autodiff::Autodiff<crate::backends::Rocm>;
        <Ad as ruda_tensor::Backend>::with_autodiff_creation_context(device, selection.context,
            || $call!(Ad, Rocm, ad, device, selection.context))
    }
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Balanced) => {
        type Ad = ruda_autodiff::Autodiff<crate::backends::Rocm, ruda_autodiff::checkpoint::strategy::BalancedCheckpointing>;
        <Ad as ruda_tensor::Backend>::with_autodiff_creation_context(device, selection.context,
            || $call!(Ad, Rocm, ad, device, selection.context))
    }
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Custom(_)) =>
        panic!("Dispatch accepts Disabled or Balanced checkpointing"),
},
#[cfg(feature = "tch")]
crate::DispatchDevice::LibTorch(device) => match selection.context {
    ruda_tensor::AutodiffContext::Disabled =>
        $call!(crate::backends::LibTorch, LibTorch, native, device, selection.context),
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Disabled) => {
        type Ad = ruda_autodiff::Autodiff<crate::backends::LibTorch>;
        <Ad as ruda_tensor::Backend>::with_autodiff_creation_context(device, selection.context,
            || $call!(Ad, LibTorch, ad, device, selection.context))
    }
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Balanced) => {
        type Ad = ruda_autodiff::Autodiff<crate::backends::LibTorch, ruda_autodiff::checkpoint::strategy::BalancedCheckpointing>;
        <Ad as ruda_tensor::Backend>::with_autodiff_creation_context(device, selection.context,
            || $call!(Ad, LibTorch, ad, device, selection.context))
    }
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Custom(_)) =>
        panic!("Dispatch accepts Disabled or Balanced checkpointing"),
},
#[cfg(feature = "remote")]
crate::DispatchDevice::Remote(device) => match selection.context {
    ruda_tensor::AutodiffContext::Disabled =>
        $call!(crate::backends::Remote, Remote, native, device, selection.context),
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Disabled) => {
        type Ad = ruda_autodiff::Autodiff<crate::backends::Remote>;
        <Ad as ruda_tensor::Backend>::with_autodiff_creation_context(device, selection.context,
            || $call!(Ad, Remote, ad, device, selection.context))
    }
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Balanced) => {
        type Ad = ruda_autodiff::Autodiff<crate::backends::Remote, ruda_autodiff::checkpoint::strategy::BalancedCheckpointing>;
        <Ad as ruda_tensor::Backend>::with_autodiff_creation_context(device, selection.context,
            || $call!(Ad, Remote, ad, device, selection.context))
    }
    ruda_tensor::AutodiffContext::Enabled(ruda_tensor::GradientCheckpointingStrategy::Custom(_)) =>
        panic!("Dispatch accepts Disabled or Balanced checkpointing"),
},
            #[cfg(feature = "capture")]
            crate::DispatchDevice::Capture(device) => {
                assert_eq!(selection.context, ruda_tensor::AutodiffContext::Disabled, "Capture does not support autodiff");
                $call!(crate::backends::Capture, Capture, native, device, selection.context)
            },
            crate::DispatchDevice::Autodiff(..) => unreachable!(),
        }
    }};
}

macro_rules! native_dispatch {
    ($device:expr, $call:ident) => {{
        match ruda_tensor::DeviceOps::inner($device) {
crate::DispatchDevice::Host(device) => $call!(crate::backends::Host, Host, device),
#[cfg(feature = "cuda")]
crate::DispatchDevice::Cuda(device) => $call!(crate::backends::Cuda, Cuda, device),
#[cfg(feature = "cpu")]
crate::DispatchDevice::Cpu(device) => $call!(crate::backends::Cpu, Cpu, device),
#[cfg(feature = "wgpu")]
crate::DispatchDevice::Wgpu(device) => $call!(crate::backends::Wgpu, Wgpu, device),
#[cfg(feature = "rocm")]
crate::DispatchDevice::Rocm(device) => $call!(crate::backends::Rocm, Rocm, device),
#[cfg(feature = "tch")]
crate::DispatchDevice::LibTorch(device) => $call!(crate::backends::LibTorch, LibTorch, device),
#[cfg(feature = "remote")]
crate::DispatchDevice::Remote(device) => $call!(crate::backends::Remote, Remote, device),
#[cfg(feature = "capture")]
crate::DispatchDevice::Capture(device) => $call!(crate::backends::Capture, Capture, device),
            crate::DispatchDevice::Autodiff(..) => unreachable!(),
        }
    }};
}

macro_rules! unpack {
(native, $variant:ident, Float, $value:expr, $context:expr) => {{
    match $value {
        crate::DispatchTensor::$variant(tensor) => tensor.into_native_float($context),
        _ => panic!("Tensor backend does not match the selected backend"),
    }
}};
(ad, $variant:ident, Float, $value:expr, $context:expr) => {{
    match $value {
        crate::DispatchTensor::$variant(tensor) => tensor.into_ad_float($context),
        _ => panic!("Tensor backend does not match the selected backend"),
    }
}};
(native, $variant:ident, Int, $value:expr, $context:expr) => {{
    match $value {
        crate::DispatchTensor::$variant(tensor) => tensor.into_native_int($context),
        _ => panic!("Tensor backend does not match the selected backend"),
    }
}};
(ad, $variant:ident, Int, $value:expr, $context:expr) => {{
    match $value {
        crate::DispatchTensor::$variant(tensor) => tensor.into_ad_int($context),
        _ => panic!("Tensor backend does not match the selected backend"),
    }
}};
(native, $variant:ident, Bool, $value:expr, $context:expr) => {{
    match $value {
        crate::DispatchTensor::$variant(tensor) => tensor.into_native_bool($context),
        _ => panic!("Tensor backend does not match the selected backend"),
    }
}};
(ad, $variant:ident, Bool, $value:expr, $context:expr) => {{
    match $value {
        crate::DispatchTensor::$variant(tensor) => tensor.into_ad_bool($context),
        _ => panic!("Tensor backend does not match the selected backend"),
    }
}};
(native, $variant:ident, Quantized, $value:expr, $context:expr) => {{
    match $value {
        crate::DispatchTensor::$variant(tensor) => tensor.into_native_quantized($context),
        _ => panic!("Tensor backend does not match the selected backend"),
    }
}};
(ad, $variant:ident, Quantized, $value:expr, $context:expr) => {{
    match $value {
        crate::DispatchTensor::$variant(tensor) => tensor.into_ad_quantized($context),
        _ => panic!("Tensor backend does not match the selected backend"),
    }
}};
}

macro_rules! pack {
(native, $variant:ident, Float, $value:expr) => {
    crate::DispatchTensor::$variant(crate::BackendTensor::Float($value))
};
(ad, $variant:ident, Float, $value:expr) => {
    crate::DispatchTensor::$variant(crate::BackendTensor::AdFloat($value))
};
(native, $variant:ident, Int, $value:expr) => {
    crate::DispatchTensor::$variant(crate::BackendTensor::Int($value))
};
(ad, $variant:ident, Int, $value:expr) => {
    crate::DispatchTensor::$variant(crate::BackendTensor::AdInt($value))
};
(native, $variant:ident, Bool, $value:expr) => {
    crate::DispatchTensor::$variant(crate::BackendTensor::Bool($value))
};
(ad, $variant:ident, Bool, $value:expr) => {
    crate::DispatchTensor::$variant(crate::BackendTensor::AdBool($value))
};
(native, $variant:ident, Quantized, $value:expr) => {
    crate::DispatchTensor::$variant(crate::BackendTensor::Quantized($value))
};
(ad, $variant:ident, Quantized, $value:expr) => {
    crate::DispatchTensor::$variant(crate::BackendTensor::AdQuantized($value))
};
}

macro_rules! borrow_tensor {
(native, $variant:ident, Float, $value:expr) => {{
    match $value {
        crate::DispatchTensor::$variant(crate::BackendTensor::Float(tensor)) => tensor,
        crate::DispatchTensor::$variant(crate::BackendTensor::AdFloat(tensor)) => &tensor.primitive,
        _ => panic!("Expected a native float tensor on the selected backend"),
    }
}};
(ad, $variant:ident, Float, $value:expr) => {{
    match $value {
        crate::DispatchTensor::$variant(crate::BackendTensor::AdFloat(tensor)) => tensor,
        _ => panic!("Expected an autodiff float tensor on the selected backend"),
    }
}};
(native, $variant:ident, Int, $value:expr) => {{
    match $value {
        crate::DispatchTensor::$variant(crate::BackendTensor::Int(tensor)) => tensor,
        crate::DispatchTensor::$variant(crate::BackendTensor::AdInt(tensor)) => &tensor.primitive,
        _ => panic!("Expected a native int tensor on the selected backend"),
    }
}};
(ad, $variant:ident, Int, $value:expr) => {{
    match $value {
        crate::DispatchTensor::$variant(crate::BackendTensor::AdInt(tensor)) => tensor,
        _ => panic!("Expected an autodiff int tensor on the selected backend"),
    }
}};
(native, $variant:ident, Bool, $value:expr) => {{
    match $value {
        crate::DispatchTensor::$variant(crate::BackendTensor::Bool(tensor)) => tensor,
        crate::DispatchTensor::$variant(crate::BackendTensor::AdBool(tensor)) => &tensor.primitive,
        _ => panic!("Expected a native bool tensor on the selected backend"),
    }
}};
(ad, $variant:ident, Bool, $value:expr) => {{
    match $value {
        crate::DispatchTensor::$variant(crate::BackendTensor::AdBool(tensor)) => tensor,
        _ => panic!("Expected an autodiff bool tensor on the selected backend"),
    }
}};
(native, $variant:ident, Quantized, $value:expr) => {{
    match $value {
        crate::DispatchTensor::$variant(crate::BackendTensor::Quantized(tensor)) => tensor,
        crate::DispatchTensor::$variant(crate::BackendTensor::AdQuantized(tensor)) => &tensor.primitive,
        _ => panic!("Expected a native quantized tensor on the selected backend"),
    }
}};
(ad, $variant:ident, Quantized, $value:expr) => {{
    match $value {
        crate::DispatchTensor::$variant(crate::BackendTensor::AdQuantized(tensor)) => tensor,
        _ => panic!("Expected an autodiff quantized tensor on the selected backend"),
    }
}};
}
