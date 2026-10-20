use ruda_model::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    tensor::{Tensor, backend::Backend},
};

/// Configuration for [`PixelShuffle`].
#[derive(Config, Debug)]
pub struct PixelShuffleConfig {
    /// Spatial upscale factor.
    pub upscale_factor: usize,
}

impl PixelShuffleConfig {
    /// Initialize the module.
    pub fn init(&self) -> PixelShuffle {
        PixelShuffle {
            upscale_factor: self.upscale_factor,
        }
    }
}

/// Rearrange `[N, C * r^2, H, W]` into `[N, C, H * r, W * r]`.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct PixelShuffle {
    /// Spatial upscale factor.
    pub upscale_factor: usize,
}

impl ModuleDisplay for PixelShuffle {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }
    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("upscale_factor", &self.upscale_factor)
            .optional()
    }
}

impl PixelShuffle {
    /// Apply the channel-to-space rearrangement to an NCHW tensor.
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, channels, height, width] = input.dims();
        let factor = self.upscale_factor;
        assert!(
            factor > 0,
            "PixelShuffle: upscale_factor must be greater than 0"
        );
        assert_eq!(
            channels % (factor * factor),
            0,
            "PixelShuffle: input channels ({channels}) must be divisible by upscale_factor^2 ({})",
            factor * factor
        );
        let out_channels = channels / (factor * factor);
        input
            .reshape([batch_size, out_channels, factor, factor, height, width])
            .permute([0, 1, 4, 2, 5, 3])
            .reshape([batch_size, out_channels, height * factor, width * factor])
    }
}

/// Configuration for [`PixelUnshuffle`].
#[derive(Config, Debug)]
pub struct PixelUnshuffleConfig {
    /// Spatial downscale factor.
    pub downscale_factor: usize,
}

impl PixelUnshuffleConfig {
    /// Initialize the module.
    pub fn init(&self) -> PixelUnshuffle {
        PixelUnshuffle {
            downscale_factor: self.downscale_factor,
        }
    }
}

/// Rearrange `[N, C, H * r, W * r]` into `[N, C * r^2, H, W]`.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct PixelUnshuffle {
    /// Spatial downscale factor.
    pub downscale_factor: usize,
}

impl ModuleDisplay for PixelUnshuffle {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }
    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("downscale_factor", &self.downscale_factor)
            .optional()
    }
}

impl PixelUnshuffle {
    /// Apply the space-to-channel rearrangement to an NCHW tensor.
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, channels, height, width] = input.dims();
        let factor = self.downscale_factor;
        assert!(
            factor > 0,
            "PixelUnshuffle: downscale_factor must be greater than 0"
        );
        assert_eq!(
            height % factor,
            0,
            "PixelUnshuffle: input height ({height}) must be divisible by downscale_factor ({factor})"
        );
        assert_eq!(
            width % factor,
            0,
            "PixelUnshuffle: input width ({width}) must be divisible by downscale_factor ({factor})"
        );
        let out_height = height / factor;
        let out_width = width / factor;
        input
            .reshape([batch_size, channels, out_height, factor, out_width, factor])
            .permute([0, 1, 3, 5, 2, 4])
            .reshape([
                batch_size,
                channels * factor * factor,
                out_height,
                out_width,
            ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use ruda_model::tensor::TensorData;
    use ruda_model::tensor::Tolerance;
    type FT = f32;

    fn sample_input() -> TensorData {
        // (1, 4, 2, 2) with values 0..16.
        TensorData::from([[
            [[0.0, 1.0], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
            [[8.0, 9.0], [10.0, 11.0]],
            [[12.0, 13.0], [14.0, 15.0]],
        ]])
    }

    #[test]
    fn pixel_shuffle_known_values() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 4>::from_data(sample_input(), &device);

        let output = PixelShuffleConfig::new(2).init().forward(input);

        // (1, 1, 4, 4); reference from a numpy pixel_shuffle implementation.
        let expected = TensorData::from([[[
            [0.0, 4.0, 1.0, 5.0],
            [8.0, 12.0, 9.0, 13.0],
            [2.0, 6.0, 3.0, 7.0],
            [10.0, 14.0, 11.0, 15.0],
        ]]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn pixel_unshuffle_inverts_pixel_shuffle() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 4>::from_data(sample_input(), &device);

        let shuffled = PixelShuffleConfig::new(2).init().forward(input.clone());
        let restored = PixelUnshuffleConfig::new(2).init().forward(shuffled);

        restored
            .to_data()
            .assert_approx_eq::<FT>(&input.to_data(), Tolerance::default());
    }

    #[test]
    fn pixel_shuffle_inverts_pixel_unshuffle() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 4>::from_data(sample_input(), &device);

        let unshuffled = PixelUnshuffleConfig::new(2).init().forward(input.clone());
        let restored = PixelShuffleConfig::new(2).init().forward(unshuffled);

        restored
            .to_data()
            .assert_approx_eq::<FT>(&input.to_data(), Tolerance::default());
    }

    #[test]
    fn pixel_shuffle_round_trip_factor_3() {
        let device = Default::default();
        // (1, 9, 1, 1): C * r^2 = 1 * 9 channels, r = 3.
        let input = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[
                [[0.0]],
                [[1.0]],
                [[2.0]],
                [[3.0]],
                [[4.0]],
                [[5.0]],
                [[6.0]],
                [[7.0]],
                [[8.0]],
            ]]),
            &device,
        );

        let shuffled = PixelShuffleConfig::new(3).init().forward(input.clone());
        let restored = PixelUnshuffleConfig::new(3).init().forward(shuffled);

        restored
            .to_data()
            .assert_approx_eq::<FT>(&input.to_data(), Tolerance::default());
    }

    #[test]
    fn display() {
        assert_eq!(
            alloc::format!("{}", PixelShuffleConfig::new(2).init()),
            "PixelShuffle {upscale_factor: 2}"
        );
        assert_eq!(
            alloc::format!("{}", PixelUnshuffleConfig::new(2).init()),
            "PixelUnshuffle {downscale_factor: 2}"
        );
    }
}
