include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tests/readback_cases.rs"
));

mod native {
    readback_cases!(crate::cuda::CudaNative);
}

#[cfg(feature = "cuda-fusion")]
mod fusion {
    readback_cases!(crate::cuda::Cuda);
}
