include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tests/ownership_cases.rs"
));
ownership_cases!(crate::LibTorch);
