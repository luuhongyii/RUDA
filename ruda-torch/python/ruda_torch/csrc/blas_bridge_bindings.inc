m.attr("blas_api_version")=1;
m.def("initialize_blas",[](uintptr_t fn){blas_native=reinterpret_cast<BlasCommand>(fn);});
m.def("blas_gemm",blas_gemm);
m.def("blas_fused",blas_fused);
m.def("blas_grouped",blas_grouped);
m.def("blas_grouped_backward",blas_grouped_backward);
m.def("blas_triangular",blas_triangular);
m.def("blas_reduce_to",blas_reduce_to);
m.def("blas_scale",blas_scale);
