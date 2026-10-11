use crate::optimizer::Optimizer;
use ruda_core::launch::RudaDim;
use ruda_core::ir::{Instruction, Processor, Scope};
use std::rc::Rc;

/// Build an optimizer with IR transformers
#[derive(Default)]
pub struct OptimizerBuilder {
    transformers: Vec<Rc<dyn IrTransformer>>,
    processors: Vec<Box<dyn Processor>>,
    pointer_values: bool,
}

impl OptimizerBuilder {
    /// Keep pointer types and operations through SSA and scalar optimization.
    pub fn with_pointer_values(mut self) -> Self {
        self.pointer_values = true;
        self
    }

    /// Add an IR transformer to the optimizer
    pub fn with_transformer(mut self, transformer: impl IrTransformer + 'static) -> Self {
        self.transformers.push(Rc::new(transformer));
        self
    }

    pub fn with_processor(mut self, processor: impl Processor + 'static) -> Self {
        self.processors.push(Box::new(processor));
        self
    }

    /// Build and run optimizer on the scope
    pub fn optimize(self, expand: Scope, ruda_dim: RudaDim) -> Optimizer {
        if self.pointer_values {
            Optimizer::new_pointer_values(expand, ruda_dim, self.transformers, self.processors)
        } else {
            Optimizer::new(expand, ruda_dim, self.transformers, self.processors)
        }
    }
}

/// The action that should be performed on an instruction, returned by [`IrTransformer::maybe_transform`]
pub enum TransformAction {
    /// The transformer doesn't apply to this instruction
    Ignore,
    /// Replace this instruction with one or more other instructions
    Replace(Vec<Instruction>),
    /// Remove this instruction with no substitute (i.e. debug info)
    Remove,
}

/// A transformer that can modify instructions before they get added to the control flow graph.
pub trait IrTransformer: core::fmt::Debug {
    /// Inspect an instruction and potentially transform it.
    fn maybe_transform(&self, scope: &mut Scope, inst: &Instruction) -> TransformAction;
}
