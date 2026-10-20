use alloc::{vec, vec::Vec};
use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

use super::{KernelArg, KernelDefinition};
use crate::ir::*;

/// Payload access of a compiled kernel argument, independent of binding mutability.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BufferIOAttr {
    /// No payload access.
    #[default]
    Dead,
    /// Reads payload.
    ReadOnly,
    /// Writes payload without reading it.
    WriteOnly,
    /// Reads and writes payload.
    ReadWrite,
}

impl BufferIOAttr {
    /// Whether the previous payload is consumed.
    pub fn reads(self) -> bool {
        matches!(self, Self::ReadOnly | Self::ReadWrite)
    }

    /// Whether the payload is produced.
    pub fn writes(self) -> bool {
        matches!(self, Self::WriteOnly | Self::ReadWrite)
    }

    fn merge(&mut self, other: Self) {
        *self = match (self.reads() || other.reads(), self.writes() || other.writes()) {
            (false, false) => Self::Dead,
            (true, false) => Self::ReadOnly,
            (false, true) => Self::WriteOnly,
            (true, true) => Self::ReadWrite,
        };
    }
}

/// Accesses in the same order as the kernel's buffer and tensor-map arguments.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelIo {
    /// Ordinary buffer arguments.
    pub buffers: Vec<BufferIOAttr>,
    /// Tensor-map backing allocations.
    pub tensor_maps: Vec<BufferIOAttr>,
}

impl KernelIo {
    /// Analyze a structured, lowered kernel without changing its compilation pipeline.
    pub fn analyze(kernel: &KernelDefinition) -> Self {
        let mut analysis = BufferIoAnalysis::new(&kernel.buffers, &kernel.tensor_maps);
        analysis.scope(&kernel.body);
        analysis.finish()
    }
}

#[derive(Clone, Default)]
struct Sources {
    indices: HashSet<usize>,
    unknown: bool,
    resolved: bool,
}

impl Sources {
    fn merge(&mut self, other: Self) {
        self.indices.extend(other.indices);
        self.unknown |= other.unknown;
        self.resolved |= other.resolved;
    }
}

/// Memory-effect analysis shared by structured and optimized compiler IR.
pub struct BufferIoAnalysis {
    buffer_ids: HashMap<Id, usize>,
    map_ids: HashMap<Id, usize>,
    buffer_count: usize,
    total: usize,
    aliases: HashMap<Variable, Vec<Variable>>,
    unknown: HashSet<Variable>,
    seeds: HashMap<Variable, Sources>,
    effects: Vec<(Variable, BufferIOAttr, bool)>,
    callbacks: Vec<(ScalarFunction, Vec<Variable>)>,
}

impl BufferIoAnalysis {
    /// Create an analysis for a kernel's argument layout.
    pub fn new(buffers: &[KernelArg], tensor_maps: &[KernelArg]) -> Self {
        Self {
            buffer_ids: buffers.iter().enumerate().map(|(i, b)| (b.id, i)).collect(),
            map_ids: tensor_maps.iter().enumerate().map(|(i, b)| (b.id, i + buffers.len())).collect(),
            buffer_count: buffers.len(),
            total: buffers.len() + tensor_maps.len(),
            aliases: HashMap::new(),
            unknown: HashSet::new(),
            seeds: HashMap::new(),
            effects: Vec::new(),
            callbacks: Vec::new(),
        }
    }

    /// Add an alias edge, including SSA phi inputs.
    pub fn alias(&mut self, out: Variable, input: Variable) {
        self.aliases.entry(out).or_default().push(input);
    }

    /// Collect all instructions in a structured scope.
    pub fn scope(&mut self, scope: &Scope) {
        for instruction in &scope.instructions {
            self.instruction(instruction);
        }
    }

    /// Collect one instruction from the IR that will be emitted.
    pub fn instruction(&mut self, instruction: &Instruction) {
        use BufferIOAttr::{ReadOnly as R, ReadWrite as RW, WriteOnly as W};
        let out = instruction.out;
        let mut aliases = Vec::new();
        let mut definition = out.is_some();
        match &instruction.operation {
            Operation::Copy(input) => aliases.push(*input),
            Operation::Operator(op) => match op {
                Operator::NativeAddress(op) => aliases.push(op.lhs),
                Operator::NativeLoad(op) => self.effect(op.input, R, true),
                Operator::NativeStore(op) => self.effect(op.lhs, W, true),
                Operator::WorkgroupUniformLoad(op) => self.effect(op.list, R, false),
                Operator::Index(op) | Operator::UncheckedIndex(op) => {
                    if out.is_some_and(|v| v.ty.is_atomic()) {
                        aliases.push(op.list);
                    } else {
                        self.effect(op.list, R, false);
                    }
                }
                Operator::IndexAssign(_) | Operator::UncheckedIndexAssign(_) => {
                    self.output(out, W);
                    definition = false;
                }
                Operator::CopyMemory(op) => {
                    self.effect(op.input, R, false);
                    self.output(out, W);
                    definition = false;
                }
                Operator::CopyMemoryBulk(op) => {
                    self.effect(op.input, R, false);
                    self.output(out, W);
                    definition = false;
                }
                Operator::Cast(op) | Operator::Reinterpret(op) => aliases.push(op.input),
                Operator::Select(op) => aliases.extend([op.then, op.or_else]),
                Operator::InitVector(_) | Operator::And(_) | Operator::Or(_) | Operator::Not(_) => {}
            },
            Operation::TypedMemory(op) => match op {
                TypedMemory::Load(pointer) => self.effect(pointer.base, R, false),
                TypedMemory::Store { pointer, .. } => self.effect(pointer.base, W, false),
            },
            Operation::Pointer(op) => match op {
                PointerOp::Address(pointer) => aliases.push(pointer.base),
                PointerOp::Offset(op) => aliases.push(op.lhs),
                PointerOp::ReadOnly(op) | PointerOp::NativeAddress(op) => aliases.push(op.input),
                PointerOp::Select(op) => aliases.extend([op.then, op.or_else]),
                PointerOp::FromNative { address, .. } => aliases.push(*address),
                PointerOp::Load { pointer, .. } | PointerOp::UniformLoad { pointer } => self.effect(*pointer, R, true),
                PointerOp::Store { pointer, .. } => self.effect(*pointer, W, true),
                PointerOp::Atomic { pointer, operation, .. } => self.effect(*pointer, match operation {
                    AtomicOpCode::Load => R,
                    AtomicOpCode::Store => W,
                    _ => RW,
                }, true),
            },
            Operation::Atomic(op) => match op {
                AtomicOp::Load(op) => self.effect(op.input, R, true),
                AtomicOp::Store(_) => {
                    if let Some(out) = out { self.effect(out, W, true); }
                    definition = false;
                }
                AtomicOp::CompareAndSwap(op) => self.effect(op.input, RW, true),
                AtomicOp::Swap(op) | AtomicOp::Add(op) | AtomicOp::Sub(op)
                | AtomicOp::Max(op) | AtomicOp::Min(op) | AtomicOp::And(op)
                | AtomicOp::Or(op) | AtomicOp::Xor(op) => self.effect(op.lhs, RW, true),
            },
            Operation::CoopMma(op) => match op {
                CoopMma::Load { value, .. } => self.effect(*value, R, false),
                CoopMma::LoadMatrix { buffer, .. } | CoopMma::LoadTensor { buffer, .. } => {
                    self.effect(*buffer, R, false);
                }
                CoopMma::Store { .. } | CoopMma::StoreMatrix { .. } | CoopMma::StoreTensor { .. } => {
                    self.output(out, W);
                    definition = false;
                }
                CoopMma::Elementwise { function, captures, .. } => {
                    self.callbacks.push(((**function).clone(), captures.clone()));
                }
                _ => {}
            },
            Operation::Barrier(op) => match op {
                BarrierOps::MemCopyAsync { source, .. }
                | BarrierOps::MemCopyAsyncCooperative { source, .. }
                | BarrierOps::MemCopyAsyncTx { source, .. }
                | BarrierOps::CopyAsync { source, .. } => {
                    self.effect(*source, R, false);
                    self.output(out, W);
                    definition = false;
                }
                BarrierOps::TmaLoad { tensor_map, .. }
                | BarrierOps::TmaLoadIm2col { tensor_map, .. } => {
                    self.effect(*tensor_map, R, false);
                    self.output(out, W);
                    definition = false;
                }
                _ => {}
            },
            Operation::Tma(TmaOps::TmaStore { source, .. }) => {
                self.effect(*source, R, false);
                self.output(out, W);
                definition = false;
            }
            Operation::Branch(branch) => match branch {
                Branch::If(b) => self.scope(&b.scope),
                Branch::IfElse(b) => { self.scope(&b.scope_if); self.scope(&b.scope_else); }
                Branch::Switch(b) => {
                    self.scope(&b.scope_default);
                    for (_, scope) in &b.cases { self.scope(scope); }
                }
                Branch::Loop(b) => self.scope(&b.scope),
                Branch::RangeLoop(b) => self.scope(&b.scope),
                _ => {}
            },
            Operation::Arithmetic(_) | Operation::Comparison(_) | Operation::Bitwise(_)
            | Operation::Metadata(_) | Operation::Synchronization(_) | Operation::Plane(_)
            | Operation::Tma(_) | Operation::NonSemantic(_) | Operation::Marker(_) => {}
        }
        if let Some(out) = out.filter(|_| definition) {
            if aliases.is_empty() {
                self.unknown.insert(out);
            } else {
                for input in aliases { self.alias(out, input); }
            }
        }
    }

    fn output(&mut self, out: Option<Variable>, access: BufferIOAttr) {
        if let Some(out) = out { self.effect(out, access, false); }
    }

    fn effect(&mut self, variable: Variable, access: BufferIOAttr, indirect: bool) {
        self.effects.push((variable, access, indirect));
    }

    fn sources(&self, variable: Variable, visited: &mut HashSet<Variable>) -> Sources {
        if let Some(seed) = self.seeds.get(&variable) { return seed.clone(); }
        let index = match variable.kind {
            VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) => {
                Some(self.buffer_ids.get(&id).copied())
            }
            VariableKind::TensorMapInput(id) | VariableKind::TensorMapOutput(id) => {
                Some(self.map_ids.get(&id).copied())
            }
            VariableKind::LocalArray { .. } | VariableKind::SharedArray { .. }
            | VariableKind::ConstantArray { .. } | VariableKind::Shared { .. }
            | VariableKind::Matrix { .. } => return Sources { resolved: true, ..Default::default() },
            _ => None,
        };
        if let Some(index) = index {
            return Sources {
                indices: index.into_iter().collect(),
                unknown: index.is_none(),
                resolved: true,
            };
        }
        if let Type::Pointer(ty) = variable.ty {
            if ty.address_space != AddressSpace::Global {
                return Sources { resolved: true, ..Default::default() };
            }
        }
        if !visited.insert(variable) { return Sources::default(); }
        let mut sources = Sources::default();
        if let Some(inputs) = self.aliases.get(&variable) {
            for input in inputs { sources.merge(self.sources(*input, visited)); }
        } else {
            sources.unknown = true;
        }
        sources.unknown |= self.unknown.contains(&variable);
        sources
    }

    fn accesses(&self) -> Vec<BufferIOAttr> {
        let mut accesses = vec![BufferIOAttr::Dead; self.total];
        for &(variable, access, indirect) in &self.effects {
            if !indirect && !matches!(variable.kind,
                VariableKind::GlobalInputArray(_) | VariableKind::GlobalOutputArray(_)
                | VariableKind::TensorMapInput(_) | VariableKind::TensorMapOutput(_))
                && !variable.ty.is_pointer()
                && !self.seeds.contains_key(&variable)
            {
                continue;
            }
            let sources = self.sources(variable, &mut HashSet::new());
            if sources.unknown || !sources.resolved {
                for slot in &mut accesses { slot.merge(access); }
            } else {
                for index in sources.indices { accesses[index].merge(access); }
            }
        }
        for (function, captures) in &self.callbacks {
            let mut child = Self {
                buffer_ids: self.buffer_ids.clone(), map_ids: self.map_ids.clone(),
                buffer_count: self.buffer_count, total: self.total,
                aliases: HashMap::new(), unknown: HashSet::new(), seeds: HashMap::new(),
                effects: Vec::new(), callbacks: Vec::new(),
            };
            for (parameter, capture) in function.body.parameters.iter()
                .skip(function.num_arguments).zip(captures)
            {
                child.seeds.insert(*parameter, self.sources(*capture, &mut HashSet::new()));
            }
            child.scope(&function.body);
            for (slot, access) in accesses.iter_mut().zip(child.accesses()) { slot.merge(access); }
        }
        accesses
    }

    /// Resolve aliases and combine effects; unattributable native access widens to all arguments.
    pub fn finish(self) -> KernelIo {
        let mut buffers = self.accesses();
        let tensor_maps = buffers.split_off(self.buffer_count);
        KernelIo { buffers, tensor_maps }
    }
}
