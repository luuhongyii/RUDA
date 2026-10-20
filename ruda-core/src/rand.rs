pub use rand::{RngExt, SeedableRng, rngs::StdRng};
pub use chacha20::ChaCha12Rng as CheckpointRng;

pub fn get_checkpoint_rng() -> CheckpointRng {
    #[cfg(feature = "std")]
    { CheckpointRng::try_from_rng(&mut rand::rngs::SysRng).unwrap() }
    #[cfg(not(feature = "std"))]
    { CheckpointRng::seed_from_u64(42) }
}

pub fn checkpoint_rng(rng: &CheckpointRng) -> alloc::vec::Vec<u8> {
    let mut record = alloc::vec::Vec::with_capacity(53);
    record.extend_from_slice(b"RC12");
    record.extend_from_slice(&rng.serialize_state());
    record
}

pub fn restore_rng(record: &[u8]) -> Result<CheckpointRng, &'static str> {
    if record.len() != 53 || &record[..4] != b"RC12" || record[52] & 0xf0 != 0 {
        return Err("invalid or incompatible ChaCha12 RNG checkpoint");
    }
    let state = record[4..].try_into().map_err(|_| "invalid RNG state size")?;
    Ok(CheckpointRng::deserialize_state(state))
}

use rand::distr::StandardUniform;
use rand::prelude::Distribution;

/// Returns a seeded random number generator using entropy.
#[cfg(feature = "std")]
#[inline(always)]
pub fn get_seeded_rng() -> StdRng {
    use rand::rngs::SysRng;

    StdRng::try_from_rng(&mut SysRng).unwrap()
}

/// Returns a seeded random number generator using a pre-generated seed.
#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn get_seeded_rng() -> StdRng {
    const CONST_SEED: u64 = 42;
    StdRng::seed_from_u64(CONST_SEED)
}

/// Generates random data from a thread-local RNG.
#[cfg(feature = "std")]
#[inline]
pub fn gen_random<T>() -> T
where
    StandardUniform: Distribution<T>,
{
    rand::rng().random()
}

/// Generates random data from a mutex-protected RNG.
#[cfg(not(feature = "std"))]
#[inline]
pub fn gen_random<T>() -> T
where
    StandardUniform: Distribution<T>,
{
    use crate::stub::Mutex;
    static RNG: Mutex<Option<StdRng>> = Mutex::new(None);
    let mut rng = RNG.lock().unwrap();
    if rng.is_none() {
        *rng = Some(get_seeded_rng());
    }
    rng.as_mut().unwrap().random()
}
