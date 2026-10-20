include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tests/ownership_cases.rs"
));

mod native {
    ownership_cases!(crate::cuda::CudaNative);
}

#[cfg(feature = "cuda-fusion")]
mod fusion {
    ownership_cases!(crate::cuda::Cuda);
}
