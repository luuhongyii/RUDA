use cudarc::{driver::{CudaContext, LaunchConfig, PushKernelArg}, nvrtc::Ptx};
use std::{error::Error, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let directory = PathBuf::from(std::env::args_os().nth(1)
        .ok_or("usage: ptx-packed <RUDA_PTX_TEST_OUTPUT directory>")?);
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = context.load_module(Ptx::from_file(directory.join("packed_fp4_copy.ptx")))?;
    let copy = module.load_function("packed_fp4_copy")?;
    for (case, length) in [0usize, 1, 63, 64, 65, 256, 257].into_iter().enumerate() {
        let started = Instant::now();
        let count = length.max(1).div_ceil(64) * 64;
        let values: Vec<u8> = (0..count + 16).map(|i| i as u8).collect();
        let input = stream.clone_htod(&values)?;
        let mut output = stream.clone_htod(&vec![0xa5u8; values.len()])?;
        let info = [length as u32; 4];
        let mut launch = stream.launch_builder(&copy);
        launch.arg(&input).arg(&mut output).arg(&info);
        // SAFETY: Buffers include every launched lane and a tail; the exported checked kernel uses this exact ABI.
        unsafe { launch.launch(LaunchConfig { grid_dim: ((count / 64) as u32, 1, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 }) }?;
        let actual = stream.clone_dtoh(&output)?;
        assert_eq!(&actual[..length], &values[..length]);
        assert!(actual[length..].iter().all(|&v| v == 0xa5));
        println!("PASS {}/8 packed-byte-copy length={length} wall_seconds={:.6}", case + 1, started.elapsed().as_secs_f64());
    }
    let module = context.load_module(Ptx::from_file(directory.join("nvfp4_fragment_coordinates.ptx")))?;
    let coordinates = module.load_function("nvfp4_fragment_coordinates")?;
    let mut output = stream.clone_htod(&vec![u32::MAX; 32 * 56 + 16])?;
    let info = [32u32 * 56; 2];
    let mut launch = stream.launch_builder(&coordinates);
    launch.arg(&mut output).arg(&info);
    // SAFETY: One full warp writes 56 coordinates per lane into the allocated output, using the exported ABI.
    unsafe { launch.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 }) }?;
    let actual = stream.clone_dtoh(&output)?;
    let mut expected = Vec::new();
    let mut seen_a = vec![0; 16 * 32];
    let mut seen_b = vec![0; 32 * 8];
    let mut seen_c = vec![0; 16 * 8];
    for lane in 0..32usize {
        let group = lane / 4;
        let thread = lane % 4;
        for register in 0..4 {
            for byte in 0..4 {
                let row = group + (register % 2) * 8;
                let col = thread * 4 + byte + (register / 2) * 16;
                expected.extend([row as u32, col as u32]);
                seen_a[row * 32 + col] += 1;
            }
        }
        for register in 0..2 {
            for byte in 0..4 {
                let row = thread * 4 + byte + register * 16;
                let col = group;
                expected.extend([row as u32, col as u32]);
                seen_b[row * 8 + col] += 1;
            }
        }
        for row in [group, group + 8] {
            for col in [thread * 2, thread * 2 + 1] {
                expected.extend([row as u32, col as u32]);
                seen_c[row * 8 + col] += 1;
            }
        }
    }
    assert_eq!(&actual[..32 * 56], expected.as_slice());
    assert!(actual[32 * 56..].iter().all(|&v| v == u32::MAX));
    assert!(seen_a.iter().chain(&seen_b).chain(&seen_c).all(|&count| count == 1));
    println!("PASS 8/8 NVFP4 packed fragment coordinates: all 32 lanes, A/B/C complete non-overlapping coverage");
    println!("NVFP4 MMA arithmetic not executed by this transport/layout test");
    Ok(())
}
