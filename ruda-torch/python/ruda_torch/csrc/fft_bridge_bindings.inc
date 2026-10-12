m.attr("fft_api_version")=1;
m.def("initialize_fft",[](uintptr_t address,uintptr_t plan_address) {
  TORCH_CHECK(allocate_native && address && plan_address && !fft_native && !fft_plan_native,
      "invalid or repeated FFT initialization");
  fft_native=reinterpret_cast<FftExecute>(address);
  fft_plan_native=reinterpret_cast<FftPlanExecute>(plan_address);
});
m.def("fft_execute",fft_bridge_call);
pybind11::class_<FftPlanBridge,std::shared_ptr<FftPlanBridge>>(m,"NativeFftPlan")
  .def(pybind11::init<const at::Tensor&,const std::vector<int64_t>&>())
  .def("execute",&FftPlanBridge::execute)
  .def("retained_bytes",&FftPlanBridge::retained_bytes)
  .def("clear_workspace",&FftPlanBridge::clear_workspace)
  .def("close",&FftPlanBridge::close);
