m.attr("vendor_api_version")=1;
m.def("initialize_vendor",[](uintptr_t address) {
  TORCH_CHECK(allocate_native && address && !vendor_scope_native,"invalid or repeated vendor initialization");
  vendor_scope_native=reinterpret_cast<VendorScope>(address);
});
m.def("vendor_execute",vendor_execute);
