/// Policy carried by an autodiff tensor for retaining or recomputing forward values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradientCheckpointingStrategy {
    /// Retain the required forward values rather than recomputing operations.
    Disabled,
    /// Recompute memory-bound operations when all required parents can be checkpointed.
    Balanced,
    /// A statically supplied backend strategy, identified by its Rust type name.
    Custom(&'static str),
}

/// Runtime autodiff association, independent of gradient retention and tape availability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutodiffContext {
    /// An ordinary constant with no graph association.
    Disabled,
    /// Autodiff is enabled with this checkpointing policy.
    Enabled(GradientCheckpointingStrategy),
}

impl AutodiffContext {
    /// Merge operation-local contexts without changing either operand.
    pub fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Disabled, context) | (context, Self::Disabled) => context,
            (Self::Enabled(left), Self::Enabled(right)) => {
                assert_eq!(left, right, "Cannot combine different gradient-checkpointing strategies");
                self
            }
        }
    }
}
