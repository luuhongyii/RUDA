#[test]
fn vector_stage_covers_rows_without_padding_or_unaligned_access() {
    for rows in [16usize, 64, 128] {
        for row_base in [0usize, 16, 64, 512, 3584] {
            let mut seen = vec![false; rows * 136];
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
    let mut output = vec![0usize; 128 * 128];
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

#[test]
fn backward_register_staging_and_output_ownership() {
    for rows in [64usize, 128] {
        let mut vector = vec![0usize; rows*136];
        let mut scalar = vec![0usize; rows*136];
        for thread in 0..256 {
            for block in 0..rows/16 {
                let row = block*16+thread/16;
                for e in 0..8 { vector[(row*17+thread%16)*8+e] += 1; }
            }
            for block in 0..rows/2 {
                let row = block*2+thread/128;
                scalar[row*136+thread%128] += 1;
            }
        }
        assert_eq!(vector,scalar);
        for (at,count) in vector.into_iter().enumerate() {
            assert_eq!(count,usize::from(at%136<128));
        }
    }
    let mut output = vec![0usize;128*128];
    let mut scores = vec![0usize;128*64];
    for thread in 0usize..256 {
        let (warp,lane) = (thread/32,thread%32);
        for j in 0..16 {
            for e in 0..4 {
                let row = warp*16+lane/4+e/2*8;
                let col = j*8+lane%4*2+e%2;
                output[row*128+col] += 1;
            }
        }
        for chunk in 0..4 {
            for part in 0..2 {
                for e in 0..4 {
                    let row = warp*16+lane/4+e/2*8;
                    let col = chunk*16+part*8+lane%4*2+e%2;
                    scores[row*64+col] += 1;
                }
            }
        }
    }
    assert!(output.into_iter().all(|n|n==1));
    assert!(scores.into_iter().all(|n|n==1));
}

#[test]
fn backward_register_head_ownership_and_mma_order() {
    for (heads,kv_heads) in [(2usize,2usize),(28,4),(14,2),(6,2)] {
        let group = heads/kv_heads;
        let kv_len = 2*kv_heads*512*128;
        for batch in 0..2 {
            for kh in 0..kv_heads {
                for h in 0..group {
                    for token in 0..512 {
                        let key = (batch*kv_heads+kh)*512+token/128*128;
                        let q = (key/(512*kv_heads)*heads+(key/512%kv_heads)*group+h)*512;
                        assert_eq!(q,(batch*heads+kh*group+h)*512);
                        let at = (key+token%128)*128+h*kv_len;
                        assert_eq!(at,((batch*kv_heads+kh)*512+token)*128+h*kv_len);
                        assert!(at+128<=group*kv_len);
                    }
                }
            }
        }
    }
    let mut contraction = Vec::new();
    for block in 0usize..8 {
        for chunk in 0..4 {
            let dot_order:Vec<_> = (0..8).map(|d|d*16).collect();
            assert_eq!(dot_order,vec![0,16,32,48,64,80,96,112]);
            contraction.push(block*64+chunk*16);
        }
    }
    assert_eq!(contraction,(0..32).map(|k|k*16).collect::<Vec<_>>());
}
