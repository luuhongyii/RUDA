class FusedAutogradDispatch final : public c10::OperatorKernel {
  c10::SafeKernelFunction original_;
 public:
  explicit FusedAutogradDispatch(c10::SafeKernelFunction original) : original_(std::move(original)) {}

  void operator()(const c10::OperatorHandle& op, c10::DispatchKeySet keys, torch::jit::Stack* stack) {
    const auto count = op.schema().arguments().size();
    TORCH_INTERNAL_ASSERT(stack->size() >= count);
    if (c10::GradMode::is_enabled()) {
      for (size_t i = stack->size() - count; i < stack->size(); ++i) {
        const auto& value = (*stack)[i];
        if (value.isTensor() && value.toTensor().requires_grad()) {
          original_.callBoxed(op, keys, stack);
          return;
        }
      }
    }
    at::AutoDispatchBelowAutograd guard;
    op.redispatchBoxed(keys & c10::after_autograd_keyset, stack);
  }
};

std::unique_ptr<torch::Library> fused_autograd_dispatch_library;

void initialize_fused_autograd_dispatch() {
  TORCH_CHECK(!fused_autograd_dispatch_library, "RUDA fused autograd dispatcher is already initialized");
  auto library = std::make_unique<torch::Library>(torch::Library::IMPL, "ruda",
      c10::DispatchKey::AutogradPrivateUse1, __FILE__, __LINE__);
  for (const char* name : {"lora_epilogue_forward", "lora_epilogue_backward", "lora_rank16_residual",
       "training_rms_forward", "training_rms_backward", "training_layer_forward", "training_layer_backward",
       "training_silu_forward", "training_silu_backward", "rotary_qk_forward", "rotary_qk_backward",
       "cross_entropy_forward", "cross_entropy_backward"}) {
    const auto op = c10::Dispatcher::singleton().findSchemaOrThrow((std::string("ruda::") + name).c_str(), "");
    auto original = op.getComputedKernelForDispatchKey(c10::DispatchKey::AutogradPrivateUse1);
    library->impl(name, torch::CppFunction::makeFromBoxedFunctor(
        std::make_unique<FusedAutogradDispatch>(std::move(original))));
  }
  fused_autograd_dispatch_library = std::move(library);
}
