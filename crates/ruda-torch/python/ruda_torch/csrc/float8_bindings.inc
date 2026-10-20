m.attr("float8_api_version")=1;
m.def("initialize_float8",[](uintptr_t address) {
  TORCH_CHECK(allocate_native && address && !float8_native,"invalid or repeated FP8 initialization");
  float8_native=reinterpret_cast<Float8Dispatch>(address);
});
m.def("float8_dispatch",float8_dispatch);
m.attr("float8_block_api_version")=1;
m.def("initialize_float8_block",[](uintptr_t address) {
  TORCH_CHECK(allocate_native && address && !float8_block_native,"invalid or repeated block FP8 initialization");
  float8_block_native=reinterpret_cast<Float8BlockDispatch>(address);
});
m.def("float8_block_dispatch",float8_block_dispatch);
m.attr("float8_shard_api_version")=1;
m.def("initialize_float8_shard",[](uintptr_t address) {
  TORCH_CHECK(allocate_native && address && !float8_shard_native,"invalid or repeated sharded FP8 initialization");
  float8_shard_native=reinterpret_cast<Float8ShardDispatch>(address);
});
m.def("float8_shard_dispatch",float8_shard_dispatch);
