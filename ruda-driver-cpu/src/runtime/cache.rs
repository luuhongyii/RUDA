pub(super) fn last_level_cache_size() -> Option<usize> {
    platform_cache_size().filter(|bytes| *bytes != 0)
}

#[cfg(target_os = "linux")]
fn platform_cache_size() -> Option<usize> {
    (0..=5)
        .filter_map(|index| {
            let path = format!("/sys/devices/system/cpu/cpu0/cache/index{index}");
            let read = |file: &str| std::fs::read_to_string(format!("{path}/{file}")).ok();
            let level: u32 = read("level")?.trim().parse().ok()?;
            (read("type")?.trim() != "Instruction").then_some(())?;
            Some((level, parse_cache_size(read("size")?.trim())?))
        })
        .max_by_key(|(level, _)| *level)
        .map(|(_, size)| size)
}

#[cfg(target_os = "linux")]
fn parse_cache_size(size: &str) -> Option<usize> {
    let (number, unit) = size.split_at(size.len().checked_sub(1)?);
    let scale = match unit {
        "K" => 1024,
        "M" => 1024 * 1024,
        _ => return None,
    };
    number.parse::<usize>().ok()?.checked_mul(scale)
}

#[cfg(target_os = "macos")]
fn platform_cache_size() -> Option<usize> {
    use std::{ffi::c_void, mem, ptr};
    let mut value = 0u64;
    let mut size = mem::size_of::<u64>();
    // Only the L3 is reported here: an Apple Silicon cluster's L2 is not its LLC.
    // SAFETY: The name is NUL-terminated and the output buffer has the advertised size.
    let result = unsafe {
        libc::sysctlbyname(
            c"hw.l3cachesize".as_ptr(), (&mut value as *mut u64).cast::<c_void>(),
            &mut size, ptr::null_mut(), 0,
        )
    };
    (result == 0 && (size == 4 || size == 8)).then_some(())?;
    usize::try_from(value).ok()
}

#[cfg(windows)]
fn platform_cache_size() -> Option<usize> {
    use std::{mem, ptr};
    use winapi::shared::minwindef::DWORD;
    use winapi::um::sysinfoapi::GetLogicalProcessorInformationEx;
    use winapi::um::winnt::{
        CacheData, CacheUnified, RelationCache, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };
    type Record = SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;

    let mut len: DWORD = 0;
    // SAFETY: A null first buffer asks Windows for the required length.
    unsafe { GetLogicalProcessorInformationEx(RelationCache, ptr::null_mut(), &mut len) };
    if len == 0 { return None; }
    let mut buffer = vec![0u64; (len as usize).div_ceil(mem::size_of::<u64>())];
    // SAFETY: u64 storage provides the record alignment and at least `len` writable bytes.
    let ok = unsafe {
        GetLogicalProcessorInformationEx(RelationCache, buffer.as_mut_ptr().cast(), &mut len)
    };
    if ok == 0 { return None; }
    let bytes = bytemuck::cast_slice::<u64, u8>(&buffer);
    let bytes = &bytes[..(len as usize).min(bytes.len())];
    let header = mem::offset_of!(Record, u);
    let mut offset = 0;
    let mut largest = None;
    while offset + header <= bytes.len() {
        let start = offset + mem::offset_of!(Record, Size);
        let size = DWORD::from_ne_bytes(
            bytes[start..start + mem::size_of::<DWORD>()].try_into().ok()?,
        ) as usize;
        if size < header || size > bytes.len() - offset { break; }
        // SAFETY: The variable-length native record is copied into an aligned,
        // zero-initialized fixed prefix. Only the reported cache variant is read.
        let mut record: Record = unsafe { mem::zeroed() };
        unsafe {
            ptr::copy_nonoverlapping(
                bytes[offset..].as_ptr(), ptr::from_mut(&mut record).cast::<u8>(),
                size.min(mem::size_of::<Record>()),
            );
        }
        if record.Relationship == RelationCache {
            let cache = unsafe { record.u.Cache() };
            if (cache.Type == CacheData || cache.Type == CacheUnified) && cache.CacheSize > 0 {
                let candidate = (cache.Level, cache.CacheSize as usize);
                largest = Some(largest.map_or(candidate, |previous| candidate.max(previous)));
            }
        }
        offset += size;
    }
    largest.map(|(_, size)| size)
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn platform_cache_size() -> Option<usize> { None }
