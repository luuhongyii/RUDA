m.attr("float8_api_version")=1;
m.def("initialize_float8",[](uintptr_t address) {
  TORCH_CHECK(allocate_native && address && !float8_native,"invalid or repeated FP8 initialization");
  float8_native=reinterpret_cast<Float8Dispatch>(address);
});
m.def("float8_dispatch",float8_dispatch);
