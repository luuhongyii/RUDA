m.attr("ring_attention_api_version") = 1;
m.attr("indexed_ring_attention_api_version") = 1;
m.def("initialize_ring_attention",[](uintptr_t address) {
  TORCH_CHECK(allocate_native && address && !ring_attention_native,"invalid or repeated ring attention initialization");
  ring_attention_native=reinterpret_cast<RingAttention>(address);
});
m.def("ring_attention",ring_attention_call);
