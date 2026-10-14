m.attr("collective_program_api_version")=1;
m.attr("replicated_tp_program_api_version")=1;
pybind11::class_<CollectiveProgram,std::shared_ptr<CollectiveProgram>>(m,"NativeCollectiveProgram")
  .def(pybind11::init<c10::intrusive_ptr<c10d::ProcessGroup>,int>())
  .def(pybind11::init<int,int,int,int,const std::vector<uintptr_t>&,uint64_t>())
  .def("run",&CollectiveProgram::run,pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("ring",&CollectiveProgram::ring_run,pybind11::call_guard<pybind11::gil_scoped_release>());
