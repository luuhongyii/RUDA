/// Calculate the expected output size when doing a pooling operation.
///
/// # Arguments
///
/// * `kernel_size` - Size of the pooling kernel
/// * `stride` - Stride of the pooling operation
/// * `padding` - Padding applied to input
/// * `dilation` - Dilation of the pooling kernel
/// * `size_in` - Input size (height or width)
/// * `ceil_mode` - If true, use ceiling instead of floor for output size calculation.
///   This allows the last pooling window to go out-of-bounds if needed.
pub fn calculate_pool_output_size(
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    size_in: usize,
    ceil_mode: bool,
) -> usize {
    let size_padded = size_in + 2 * padding;
    let kernel_extent = dilation * (kernel_size - 1) + 1;
    if ceil_mode {
        let size_out = (size_padded + stride - 1 - kernel_extent) / stride + 1;
        if (size_out - 1) * stride >= size_in + padding {
            size_out - 1
        } else {
            size_out
        }
    } else {
        // Floor division (default)
        (size_padded - kernel_extent) / stride + 1
    }
}

#[cfg(test)]
mod tests {
    use super::calculate_pool_output_size as size;

    #[test]
    fn ceil_pool_excludes_windows_starting_after_input() {
        assert_eq!(size(2, 2, 1, 1, 7, true), 4);
        assert_eq!(size(2, 3, 1, 2, 5, true), 2);
        assert_eq!(size(1, 3, 0, 1, 5, true), 2);
        assert_eq!(size(3, 2, 1, 1, 6, true), 4);
        assert_eq!(size(3, 2, 0, 1, 2, true), 1);
    }
}

