m.attr("scientific_api_version")=1;
m.def("initialize_scientific",[](uintptr_t address) {
  TORCH_CHECK(allocate_native && address && !scientific_native,"invalid or repeated scientific initialization");
  scientific_native=reinterpret_cast<Scientific>(address);
});
m.def("scientific_call",scientific_call);
m.def("scientific_resolve",scientific_resolve);
