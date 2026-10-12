m.attr("linalg_api_version")=1;
m.def("initialize_linalg",[](uintptr_t address){TORCH_CHECK(allocate_native&&address&&!linalg_bridge_native,"invalid/repeated linalg initialization");linalg_bridge_native=reinterpret_cast<LinalgBridge>(address);});
m.def("linalg",linalg_bridge_call);
