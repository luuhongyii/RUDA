m.attr("dense_attention_api_version") = 1;
m.def("initialize_dense_attention",[](uintptr_t address) {
  TORCH_CHECK(allocate_native && address && !dense_attention_native,"invalid or repeated dense attention initialization");
  dense_attention_native=reinterpret_cast<DenseAttention>(address);
});
m.def("dense_attention",dense_attention_call);
