#![allow(unknown_lints, unnecessary_transmutes)]

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use ruda_core::kernel::Visibility;
use crate::optimizer::Optimizer;
use rspirv::{binary::Disassemble, dr::Module};

mod arithmetic;
mod atomic;
mod bitwise;
mod branch;
mod cmma;
mod compiler;
mod debug;
mod extensions;
mod extended;
mod function;
mod globals;
mod instruction;
mod item;
mod lookups;
mod metadata;
mod subgroup;
mod sync;
mod tensor_addressing;
mod target;
mod variable;
mod uniform_load;

pub use compiler::*;
use serde::{Deserialize, Serialize};
pub use target::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpirvKernel {
    #[serde(default)]
    pub io: Option<ruda_core::kernel::KernelIo>,
    #[serde(skip)]
    pub module: Option<Arc<Module>>,
    #[serde(skip)]
    pub optimizer: Option<Arc<SpirvOptimizerSnapshot>>,

    pub assembled_module: Vec<u32>,
    pub bindings: Vec<Visibility>,
    pub shared_size: usize,
    pub uniform_info: bool,
}

#[derive(Debug, Clone)]
pub struct SpirvOptimizerSnapshot {
    ir: String,
    dot: String,
}

impl SpirvOptimizerSnapshot {
    fn new(optimizer: &Optimizer) -> Self {
        Self { ir: optimizer.to_string(), dot: optimizer.dot_viz().to_string() }
    }

    pub fn dot_viz(&self) -> &str { &self.dot }
}

impl Display for SpirvOptimizerSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.ir)
    }
}

impl Eq for SpirvKernel {}
impl PartialEq for SpirvKernel {
    fn eq(&self, other: &Self) -> bool {
        self.assembled_module == other.assembled_module
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpirvCacheEntry {
    pub entrypoint_name: String,
    pub kernel: SpirvKernel,
}

impl SpirvCacheEntry {
    pub fn new(entrypoint_name: String, kernel: SpirvKernel) -> Self {
        SpirvCacheEntry {
            entrypoint_name,
            kernel,
        }
    }
}

impl Display for SpirvKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(module) = &self.module {
            write!(f, "{}", module.disassemble())
        } else {
            f.write_str("SPIR-V")
        }
    }
}

#[derive(Clone, Copy)]
pub struct SpirvLowering {
    pub debug_symbols: fn() -> bool,
    pub optimize: fn(&ruda_core::kernel::KernelDefinition, ruda_core::launch::ExecutionMode, &ruda_core::compiler::WgpuCompilationOptions) -> Optimizer,
}
