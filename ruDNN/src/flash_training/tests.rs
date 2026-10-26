#[test]
fn vector_stage_covers_rows_without_padding_or_unaligned_access() {
    for rows in [16usize, 64, 128] {
        for row_base in [0usize, 16, 64, 512, 3584] {
            let mut seen = alloc::vec![false; rows * 136];
            for block in 0..rows / 8 {
                for thread in 0..128 {
                    let row = block * 8 + thread / 16;
                    let target = (row * 17 + thread % 16) * 8;
                    let source = ((row_base + row) * 16 + thread % 16) * 8;
                    assert_eq!(target % 8, 0);
                    assert_eq!(source % 8, 0);
                    for element in 0..8 {
                        let at = target + element;
                        assert!(!seen[at]);
                        seen[at] = true;
                        assert_eq!(source + element, (row_base + at / 136) * 128 + at % 136);
                    }
                }
            }
            for (at, touched) in seen.into_iter().enumerate() {
                assert_eq!(touched, at % 136 < 128);
            }
        }
    }
}

#[test]
fn adjacent_accumulators_match_bf16_mma_a_fragment_coordinates() {
    for lane in 0usize..32 {
        for element in 0..8 {
            let a_row = lane / 4 + if element % 4 < 2 { 0 } else { 8 };
            let a_col = (lane % 4) * 2 + element % 2 + if element < 4 { 0 } else { 8 };
            let part = element / 4;
            let c = element % 4;
            let c_row = lane / 4 + (c / 2) * 8;
            let c_col = (lane % 4) * 2 + c % 2;
            assert_eq!((a_row, a_col), (c_row, part * 8 + c_col));
        }
    }
}

#[test]
fn register_rows_preserve_softmax_order_and_output_ownership() {
    let mut output = alloc::vec![0usize; 128 * 128];
    let mut stats = [0usize; 128];
    for thread in 0usize..128 {
        let (warp, lane) = (thread / 32, thread % 32);
        for i in 0..2 {
            for r in 0..2 {
                let row = warp * 32 + i * 16 + lane / 4 + r * 8;
                if lane % 4 == 0 { stats[row] += 1; }
                for p in 0..4 {
                    let source_lane = lane / 4 * 4 + p;
                    assert_eq!(source_lane / 4, lane / 4);
                }
                for element in 0..16 {
                    let original_col = element / 2 * 8 + lane % 4 * 2 + element % 2;
                    let fragment = element / 2;
                    let component = element % 2;
                    assert_eq!(original_col, fragment * 8 + lane % 4 * 2 + component);
                }
                for j in 0..16 {
                    for c in 0..2 { output[row * 128 + j * 8 + lane % 4 * 2 + c] += 1; }
                }
            }
        }
    }
    assert!(stats.into_iter().all(|n| n == 1));
    assert!(output.into_iter().all(|n| n == 1));
}

#[test]
fn register_rows_preserve_causal_block_traversal_and_head_mapping() {
    for batch in 0usize..3 {
        for (heads, kv_heads) in [(2usize, 2usize), (28, 4), (14, 2), (6, 2)] {
            for head in 0..heads {
                for row in 0..512 {
                    let original = (batch * heads + head) * 512 + row / 16 * 16;
                    let register = (batch * heads + head) * 512 + row / 128 * 128;
                    let blocks = (register % 512 / 128 + 1) * 2;
                    assert_eq!(blocks, (original % 512 / 128 + 1) * 2);
                    let kv = (register / (heads * 512) * kv_heads
                        + (register / 512 % heads) / (heads / kv_heads)) * 512;
                    assert_eq!(kv, (batch * kv_heads + head / (heads / kv_heads)) * 512);
                    for step in 0..blocks {
                        assert_eq!(blocks - 1 - step, (original % 512 / 128 + 1) * 2 - 1 - step);
                    }
                }
            }
        }
    }
}
