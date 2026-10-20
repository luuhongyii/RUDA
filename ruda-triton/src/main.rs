use std::io::Read;

fn main() {
    let mut input = String::new();
    if let Err(error) = std::io::stdin().read_to_string(&mut input) {
        eprintln!("failed to read SIMT kernel: {error}");
        std::process::exit(2);
    }
    let result = serde_json::from_str::<ruda_triton::Kernel>(&input)
        .map_err(|e| format!("invalid SIMT kernel: {e}"))
        .and_then(|kernel| ruda_triton::compile(&kernel));
    match result {
        Ok(compiled) => println!("{}", serde_json::to_string(&compiled).unwrap()),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
