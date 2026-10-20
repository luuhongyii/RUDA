m.attr("varlen_attention_api_version") = 1;
m.def("initialize_varlen_attention",[](uintptr_t address) {
  TORCH_CHECK(allocate_native && address && !varlen_attention_native,"invalid or repeated varlen initialization");
  varlen_attention_native=reinterpret_cast<VarlenAttention>(address);
});
m.def("varlen_attention",varlen_attention_call);
