use ruda::runtime::tune::{LocalTuner, Tunable, TunableSet};
use ruda_kernel::{dsl as kernel_dsl, dsl::{AutotuneKey, Runtime, RudaTuneId, ir::StorageType}, tensor::RudaTensor};
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
struct TileKey {
    k: usize,
    n: usize,
    lda: usize,
    ldb: usize,
    nt: bool,
    output: StorageType,
}

fn key<R: Runtime>((a, b, out): &(RudaTensor<R>, RudaTensor<R>, RudaTensor<R>)) -> TileKey {
    let nt = b.meta.strides()[0] == 1;
    TileKey { k: a.meta.shape()[1], n: b.meta.shape()[1], lda: a.meta.strides()[0],
        ldb: if nt { b.meta.strides()[1] } else { b.meta.strides()[0] }, nt, output: out.dtype.into() }
}

fn inputs<R: Runtime>(_: &TileKey,
    (a, b, out): &(RudaTensor<R>, RudaTensor<R>, RudaTensor<R>),
) -> (RudaTensor<R>, RudaTensor<R>, RudaTensor<R>) {
    let mut scratch = out.clone();
    let mut handle = out.client.empty(usize::try_from(out.handle.size()).expect("training matmul span exceeds usize"));
    handle.offset_start = out.handle.offset_start;
    handle.offset_end = out.handle.offset_end;
    scratch.handle = handle;
    (a.clone(), b.clone(), scratch)
}

pub(super) fn matmul<R: Runtime>(a: RudaTensor<R>, b: RudaTensor<R>, out: RudaTensor<R>) {
    static TUNER: LocalTuner<TileKey, RudaTuneId> = LocalTuner::new("rublas::training_bf16::ordered_tiles::v1");
    let client = a.client.clone();
    let id = RudaTuneId::new(&client, &a.device);
    let candidates = TUNER.init_for_device(&id, || {
        let mut set = TunableSet::new(key::<R>, inputs::<R>);
        for (name, m, n, k, asynchronous) in [
            ("sync_128x128x32", 128, 128, 32, false),
            ("async_64x64x32", 64, 64, 32, true),
            ("async_128x64x32", 128, 64, 32, true),
            ("async_128x128x32", 128, 128, 32, true),
            ("async_128x64x64", 128, 64, 64, true),
            ("async_128x128x64", 128, 128, 64, true),
        ] {
            set = set.with(Tunable::new(name,
                move |(a, b, out): (RudaTensor<R>, RudaTensor<R>, RudaTensor<R>)| {
                    super::training_gemm::launch_tile(a, b, out, m, n, k, asynchronous)
                }));
        }
        set
    });
    TUNER.execute(&id, &client, candidates, (a, b, out));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_tile_keys_do_not_bucket_k_or_strides() {
        let a = TileKey { k: 96, n: 256, lda: 128, ldb: 256, nt: false,
            output: ruda_core::tensor::DType::F32.into() };
        assert_ne!(a, TileKey { k: 128, ..a.clone() });
        assert_ne!(a, TileKey { lda: 256, ..a.clone() });
        assert_ne!(a, TileKey { ldb: 512, ..a.clone() });
        assert_ne!(a, TileKey { nt: true, ..a.clone() });
        assert_ne!(a, TileKey { output: ruda_core::tensor::DType::BF16.into(), ..a.clone() });
    }
}
