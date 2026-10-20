use alloc::{string::String, vec::Vec};
use crate::{ir::{Id, Scope, StorageType, Type}, launch::RudaDim};
use serde::{Deserialize, Serialize};

mod io;
pub use io::{BufferIOAttr, BufferIoAnalysis, KernelIo};

#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct KernelDefinition {
    pub buffers: Vec<KernelArg>,
    pub tensor_maps: Vec<KernelArg>,
    pub scalars: Vec<ScalarKernelArg>,
    pub ruda_dim: RudaDim,
    pub body: Scope,
    pub options: KernelOptions,
}

#[derive(Default, Clone, Debug, Hash, PartialEq, Eq)]
/// Options for a specific kernel compilation
pub struct KernelOptions {
    /// The name of the kernel
    pub kernel_name: String,
    /// Whether to include debug symbols
    pub debug_symbols: bool,
    /// CUDA Cluster dim, if any
    pub cluster_dim: Option<RudaDim>,
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
/// Global argument of a kernel.
pub struct KernelArg {
    /// The kernel id.
    pub id: Id,
    /// Whether the global argument can only be accessed for reading, or if it can also be accessed
    /// for write.
    pub visibility: Visibility,
    /// The type of the argument.
    pub ty: Type,
    /// The size of the argument.
    pub size: Option<usize>,
    /// Whether the argument has metadata.
    pub has_extended_meta: bool,
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ScalarKernelArg {
    pub ty: StorageType,
    pub count: usize,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum Visibility {
    Read,
    ReadWrite,
}

impl KernelDefinition {
    pub fn validate_typed_memory(&self) -> Result<(), String> {
        fn visit(kernel: &KernelDefinition, scope: &Scope) -> Result<(), String> {
            use crate::ir::{Branch, Operation, PointerAccess, VariableKind};
            for instruction in &scope.instructions {
                match &instruction.operation {
                    Operation::CoopMma(crate::ir::CoopMma::Elementwise { function, .. }) => {
                        visit(kernel, &function.body)?;
                    }
                    Operation::Pointer(op) => {
                        op.validate(instruction.out)?;
                        if let crate::ir::PointerOp::Address(pointer) = op {
                            if let VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) = pointer.base.kind {
                                let binding = kernel.buffers.iter().find(|binding| binding.id == id)
                                    .ok_or_else(|| alloc::format!("pointer value refers to unknown buffer {id}"))?;
                                if binding.ty != pointer.base.ty { return Err("pointer value buffer type mismatch".into()); }
                                if pointer.ty.access == PointerAccess::ReadWrite && binding.visibility != Visibility::ReadWrite {
                                    return Err("read-only binding cannot provide a writable pointer value".into());
                                }
                            }
                        }
                    }
                    Operation::TypedMemory(op) => {
                        op.validate(instruction.out)?;
                        let pointer = op.pointer();
                        if let VariableKind::GlobalInputArray(id)
                        | VariableKind::GlobalOutputArray(id) = pointer.base.kind
                        {
                            let binding = kernel
                                .buffers
                                .iter()
                                .find(|binding| binding.id == id)
                                .ok_or_else(|| {
                                    alloc::format!("typed pointer refers to unknown buffer {id}")
                                })?;
                            if binding.ty != pointer.base.ty {
                                return Err(
                                    "typed pointer does not match its buffer element type".into()
                                );
                            }
                            if pointer.ty.access == PointerAccess::ReadWrite
                                && binding.visibility != Visibility::ReadWrite
                            {
                                return Err("read-only kernel binding cannot provide a writable typed pointer".into());
                            }
                        }
                    }
                    Operation::Branch(branch) => match branch {
                        Branch::If(branch) => visit(kernel, &branch.scope)?,
                        Branch::IfElse(branch) => {
                            visit(kernel, &branch.scope_if)?;
                            visit(kernel, &branch.scope_else)?;
                        }
                        Branch::Switch(branch) => {
                            visit(kernel, &branch.scope_default)?;
                            for (_, scope) in &branch.cases {
                                visit(kernel, scope)?;
                            }
                        }
                        Branch::RangeLoop(branch) => visit(kernel, &branch.scope)?,
                        Branch::Loop(branch) => visit(kernel, &branch.scope)?,
                        _ => {}
                    },
                    _ => {}
                }
            }
            Ok(())
        }
        let mut body = self.body.clone();
        body.lower_reference_values()?;
        visit(self, &body)
    }
}
