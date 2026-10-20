use ruda_model::config::Config;
use ruda_model::module::{Content, DisplaySettings, Flag, Module, ModuleDisplay, Param};
use ruda_model::tensor::{Distribution, Tensor, activation::leaky_relu, backend::Backend};

/// Randomized leaky ReLU configuration.
#[derive(Config, Debug)]
pub struct RReluConfig {
    /// Lower bound of the sampled negative slope.
    #[config(default = 0.125)]
    pub lower: f64,
    /// Upper bound of the sampled negative slope.
    #[config(default = "1.0 / 3.0")]
    pub upper: f64,
}

/// Randomized leaky ReLU; evaluation uses the midpoint negative slope.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct RRelu {
    /// Lower bound of the sampled negative slope.
    pub lower: f64,
    /// Upper bound of the sampled negative slope.
    pub upper: f64,
    /// Permit stochastic behavior on an autodiff device.
    pub training: Param<Flag>,
}

impl RReluConfig {
    /// Initialize the activation.
    pub fn init(&self) -> RRelu {
        assert!(self.lower <= self.upper, "RRelu requires lower <= upper");
        RRelu {
            lower: self.lower,
            upper: self.upper,
            training: Param::from_bool(true),
        }
    }
}

impl RRelu {
    /// Apply the activation elementwise, preserving shape.
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        if !self.training.is_enabled()
            || !input.is_autodiff()
            || self.lower == self.upper
        {
            return leaky_relu(input, (self.lower + self.upper) / 2.0);
        }
        let negative = input.clone().lower_elem(0.0);
        let slopes = input.random_like(Distribution::Uniform(self.lower, self.upper));
        let scaled = input.clone() * slopes;
        input.mask_where(negative, scaled)
    }
}

impl ModuleDisplay for RRelu {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }
    fn custom_content(&self, content: Content) -> Option<Content> {
        let content = content.add("lower", &self.lower).add("upper", &self.upper);
        if self.training.is_enabled() {
            content.optional()
        } else {
            content.add("training", &self.training).optional()
        }
    }
}
