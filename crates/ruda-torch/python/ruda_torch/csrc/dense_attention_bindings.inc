m.attr("dense_attention_api_version") = 1;
m.def("initialize_dense_attention",[](uintptr_t address) {
  TORCH_CHECK(allocate_native && address && !dense_attention_native,"invalid or repeated dense attention initialization");
  dense_attention_native=reinterpret_cast<DenseAttention>(address);
});
m.def("dense_attention",dense_attention_call);
m.attr("indexed_dense_attention_api_version") = 1;
m.def("initialize_indexed_dense_attention",[](uintptr_t address) {
  TORCH_CHECK(allocate_native && address && !indexed_dense_attention_native,"invalid or repeated indexed dense attention initialization");
  indexed_dense_attention_native=reinterpret_cast<IndexedDenseAttention>(address);
});
m.def("indexed_dense_attention",indexed_dense_attention_call);
m.attr("attention_dropout_api_version") = 1;
m.def("initialize_attention_dropout",[](uintptr_t address) {
  TORCH_CHECK(allocate_native && address && !attention_dropout_native,"invalid or repeated attention dropout initialization");
  attention_dropout_native=reinterpret_cast<AttentionDropout>(address);
});
m.def("attention_dropout",attention_dropout_call);
