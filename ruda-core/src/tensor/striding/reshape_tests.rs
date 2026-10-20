use super::*;

#[test]
fn test_reshape_analysis_is_contiguous() {
    let analysis = reshape_analysis(
        &[32, 1, 1, 1].into(),
        Some(&[1, 1, 1, 1].into()),
        &[1, 1, 32, 1, 1, 1].into(),
    );

    assert_eq!(analysis, ReshapeAnalysis::IsContiguous)
}

#[test]
fn test_reshape_analysis_is_contiguous_2() {
    let analysis = reshape_analysis(
        &[32, 1, 1, 8].into(),
        Some(&[8, 8, 8, 1].into()),
        &[1, 1, 32, 1, 1, 8].into(),
    );

    assert_eq!(analysis, ReshapeAnalysis::IsContiguous)
}

#[test]
fn test_reshape_analysis_broadcasted_batch() {
    let analysis = reshape_analysis(
        &[32, 1, 1, 1].into(),
        Some(&[1, 32, 32, 32].into()),
        &[1, 1, 32, 1, 1, 1].into(),
    );

    assert_eq!(analysis, ReshapeAnalysis::Broadcasted)
}

#[test]
fn test_reshape_analysis_unsqueeze_split() {
    // Unsqueeze
    let analysis = reshape_analysis(
        &[32, 1, 1, 1].into(),
        Some(&[1, 32, 32, 32].into()),
        &[32, 1, 1, 1, 1].into(),
    );

    assert_eq!(analysis, ReshapeAnalysis::Split)
}

#[test]
fn test_reshape_analysis_split() {
    let analysis = reshape_analysis(
        &[32, 1, 1, 1].into(),
        Some(&[1, 32, 32, 32].into()),
        &[4, 8, 1, 1, 1].into(),
    );

    assert_eq!(analysis, ReshapeAnalysis::Split)
}

#[test]
fn split_with_original_singleton_axes_preserves_offsets() {
    fn offset(mut linear: usize, shape: &[usize], strides: &[usize]) -> usize {
        let mut offset = 0;
        for (&dim, &stride) in shape.iter().zip(strides).rev() {
            offset += (linear % dim) * stride;
            linear /= dim;
        }
        offset
    }
    for (shape, strides, target) in [
        (&[3, 1][..], &[1, 128][..], &[1, 3, 1, 1][..]),
        (&[1, 3][..], &[128, 1][..], &[3, 1, 1][..]),
        (&[2, 1, 3][..], &[128, 128, 1][..], &[1, 2, 1, 3, 1][..]),
        (&[1, 6, 1, 4, 1][..], &[768, 128, 128, 1, 1][..], &[2, 3, 1, 2, 2, 1][..]),
    ] {
        let ReshapeAction::UpdateStrides { strides: new_strides } =
            reshape_action(&shape.into(), &strides.into(), &target.into()) else {
                panic!("Expected a stride-only reshape");
            };
        for linear in 0..shape.iter().product() {
            assert_eq!(offset(linear, shape, strides), offset(linear, target, &new_strides));
        }
    }
}
