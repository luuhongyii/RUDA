include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../ruda-tensor/tests/ownership_cases.rs"
));
ownership_cases!(crate::LibTorch);
