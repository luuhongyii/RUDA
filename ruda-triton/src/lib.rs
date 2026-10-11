//! Lowers a SIMT kernel description produced by the `ruda_triton` Python backend into Ruda IR
//! and compiles it with the direct PTX backend.
//!
//! The description is already scalarized: every value is a scalar register, a constant, a
//! builtin, a scalar kernel argument or the base address of a pointer argument. Tensor layout,
//! reductions and data exchange are resolved by the Python lowering.

use ruda_compiler::ptx::{PtxCompilationOptions, PtxCompiler, PtxTarget};
use ruda_core::{
    compiler::Compiler,
    ir::*,
    kernel::{KernelArg, KernelDefinition, KernelOptions, ScalarKernelArg, Visibility},
    launch::{ExecutionMode, RudaDim},
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct Kernel {
    pub name: String,
    pub threads: u32,
    pub sm: u32,
    pub ptx_version: (u32, u32),
    pub num_ptrs: u32,
    pub scalars: Vec<String>,
    #[serde(default)]
    pub shared: Vec<Shared>,
    pub locals: Vec<String>,
    pub body: Vec<Stmt>,
}

#[derive(Deserialize)]
pub struct Shared {
    pub ty: String,
    pub len: usize,
}

#[derive(Deserialize, Clone)]
#[serde(untagged)]
pub enum Operand {
    Local { l: u32 },
    Const { c: String, t: String },
    Builtin { b: String },
    Scalar { s: usize },
    Ptr { p: u32 },
}

#[derive(Deserialize)]
#[serde(tag = "k", rename_all = "lowercase")]
pub enum Stmt {
    Bin { op: String, out: u32, a: Operand, b: Operand },
    Un { op: String, out: u32, a: Operand },
    Fma { out: u32, a: Operand, b: Operand, c: Operand },
    Cmp { op: String, out: u32, a: Operand, b: Operand },
    Cast { out: u32, a: Operand },
    Bitcast { out: u32, a: Operand },
    Sel { out: u32, c: Operand, a: Operand, b: Operand },
    Mov { out: u32, a: Operand },
    Ld { out: u32, ptr: Operand },
    St { ptr: Operand, val: Operand },
    Shld { out: u32, sh: usize, idx: Operand },
    Shst { sh: usize, idx: Operand, val: Operand },
    Bar,
    If {
        c: Operand,
        then: Vec<Stmt>,
        #[serde(rename = "else", default)]
        otherwise: Vec<Stmt>,
    },
    Loop { body: Vec<Stmt> },
    Brk,
}

#[derive(Serialize)]
pub struct Compiled {
    pub name: String,
    pub ptx: String,
    pub threads: u32,
    pub shared: usize,
    pub info_size: usize,
    pub scalar_offsets: Vec<usize>,
}

pub fn parse_type(name: &str) -> Result<Type, String> {
    let elem = match name {
        "f16" => ElemType::Float(FloatKind::F16),
        "bf16" => ElemType::Float(FloatKind::BF16),
        "f32" => ElemType::Float(FloatKind::F32),
        "f64" => ElemType::Float(FloatKind::F64),
        "i1" => ElemType::Bool,
        "i8" => ElemType::Int(IntKind::I8),
        "i16" => ElemType::Int(IntKind::I16),
        "i32" => ElemType::Int(IntKind::I32),
        "i64" => ElemType::Int(IntKind::I64),
        "u8" => ElemType::UInt(UIntKind::U8),
        "u16" => ElemType::UInt(UIntKind::U16),
        "u32" => ElemType::UInt(UIntKind::U32),
        "u64" => ElemType::UInt(UIntKind::U64),
        other => return Err(format!("unsupported SIMT type {other}")),
    };
    Ok(Type::scalar(elem))
}

fn parse_constant(text: &str, ty: Type) -> Result<ConstantValue, String> {
    let bad = |e: &dyn std::fmt::Display| format!("constant {text:?}: {e}");
    let elem = ty.elem_type();
    Ok(match elem {
        ElemType::Float(_) => {
            let bits = u64::from_str_radix(text.trim_start_matches("0x"), 16).map_err(|e| bad(&e))?;
            ConstantValue::Float(f64::from_bits(bits))
        }
        ElemType::Int(_) => ConstantValue::Int(text.parse().map_err(|e| bad(&e))?),
        ElemType::UInt(_) => ConstantValue::UInt(text.parse().map_err(|e| bad(&e))?),
        ElemType::Bool => ConstantValue::Bool(text == "1" || text == "true"),
    })
}

struct Builder {
    locals: Vec<Type>,
    scalars: Vec<(Type, u32)>,
    ptr_bases: Vec<Variable>,
    shared: Vec<Variable>,
}

impl Builder {
    fn local(&self, id: u32) -> Result<Variable, String> {
        let ty = *self.locals.get(id as usize).ok_or_else(|| format!("unknown local {id}"))?;
        Ok(Variable::new(VariableKind::LocalMut { id }, ty))
    }

    fn operand(&self, op: &Operand) -> Result<Variable, String> {
        Ok(match op {
            Operand::Local { l } => self.local(*l)?,
            Operand::Const { c, t } => {
                let ty = parse_type(t)?;
                Variable::constant(parse_constant(c, ty)?, ty)
            }
            Operand::Builtin { b } => {
                let builtin = match b.as_str() {
                    "tid" => Builtin::UnitPosX,
                    "pidx" => Builtin::RudaPosX,
                    "pidy" => Builtin::RudaPosY,
                    "pidz" => Builtin::RudaPosZ,
                    "npx" => Builtin::RudaCountX,
                    "npy" => Builtin::RudaCountY,
                    "npz" => Builtin::RudaCountZ,
                    other => return Err(format!("unknown builtin {other}")),
                };
                Variable::builtin(builtin, UIntKind::U32.into())
            }
            Operand::Scalar { s } => {
                let (ty, id) = *self.scalars.get(*s).ok_or_else(|| format!("unknown scalar {s}"))?;
                Variable::new(VariableKind::GlobalScalar(id), ty)
            }
            Operand::Ptr { p } => *self
                .ptr_bases
                .get(*p as usize)
                .ok_or_else(|| format!("unknown pointer argument {p}"))?,
        })
    }

    fn block(&self, parent: &mut Scope, stmts: &[Stmt]) -> Result<Scope, String> {
        let mut scope = parent.child();
        for stmt in stmts {
            self.stmt(&mut scope, stmt)?;
        }
        Ok(scope)
    }

    fn stmt(&self, scope: &mut Scope, stmt: &Stmt) -> Result<(), String> {
        let push = |scope: &mut Scope, op: Operation, out: Variable| {
            scope.instructions.push(Instruction::new(op, out));
        };
        match stmt {
            Stmt::Bin { op, out, a, b } => {
                let out = self.local(*out)?;
                let bin = BinaryOperator { lhs: self.operand(a)?, rhs: self.operand(b)? };
                let operation: Operation = match op.as_str() {
                    "add" => Arithmetic::Add(bin).into(),
                    "sub" => Arithmetic::Sub(bin).into(),
                    "mul" => Arithmetic::Mul(bin).into(),
                    "div" => Arithmetic::Div(bin).into(),
                    "rem" => Arithmetic::Remainder(bin).into(),
                    "mod" => Arithmetic::Modulo(bin).into(),
                    "max" => Arithmetic::Max(bin).into(),
                    "min" => Arithmetic::Min(bin).into(),
                    "pow" => Arithmetic::Powf(bin).into(),
                    "mulhi" => Arithmetic::MulHi(bin).into(),
                    "and" => Bitwise::BitwiseAnd(bin).into(),
                    "or" => Bitwise::BitwiseOr(bin).into(),
                    "xor" => Bitwise::BitwiseXor(bin).into(),
                    "shl" => Bitwise::ShiftLeft(bin).into(),
                    "shr" => Bitwise::ShiftRight(bin).into(),
                    other => return Err(format!("unknown binary op {other}")),
                };
                push(scope, operation, out);
            }
            Stmt::Un { op, out, a } => {
                let out = self.local(*out)?;
                let un = UnaryOperator { input: self.operand(a)? };
                let operation: Operation = match op.as_str() {
                    "neg" => Arithmetic::Neg(un).into(),
                    "abs" => Arithmetic::Abs(un).into(),
                    "exp" => Arithmetic::Exp(un).into(),
                    "log" => Arithmetic::Log(un).into(),
                    "log1p" => Arithmetic::Log1p(un).into(),
                    "sqrt" => Arithmetic::Sqrt(un).into(),
                    "rsqrt" => Arithmetic::InverseSqrt(un).into(),
                    "sin" => Arithmetic::Sin(un).into(),
                    "cos" => Arithmetic::Cos(un).into(),
                    "tanh" => Arithmetic::Tanh(un).into(),
                    "erf" => Arithmetic::Erf(un).into(),
                    "floor" => Arithmetic::Floor(un).into(),
                    "ceil" => Arithmetic::Ceil(un).into(),
                    "round" => Arithmetic::Round(un).into(),
                    "trunc" => Arithmetic::Trunc(un).into(),
                    "recip" => Arithmetic::Recip(un).into(),
                    "not" => Bitwise::BitwiseNot(un).into(),
                    other => return Err(format!("unknown unary op {other}")),
                };
                push(scope, operation, out);
            }
            Stmt::Fma { out, a, b, c } => {
                let out = self.local(*out)?;
                let fma = FmaOperator { a: self.operand(a)?, b: self.operand(b)?, c: self.operand(c)? };
                push(scope, Arithmetic::Fma(fma).into(), out);
            }
            Stmt::Cmp { op, out, a, b } => {
                let out = self.local(*out)?;
                let lhs = self.operand(a)?;
                let bin = BinaryOperator { lhs, rhs: self.operand(b)? };
                let operation: Operation = match op.as_str() {
                    "lt" => Comparison::Lower(bin).into(),
                    "le" => Comparison::LowerEqual(bin).into(),
                    "gt" => Comparison::Greater(bin).into(),
                    "ge" => Comparison::GreaterEqual(bin).into(),
                    "eq" => Comparison::Equal(bin).into(),
                    "ne" => Comparison::NotEqual(bin).into(),
                    "isnan" => Comparison::IsNan(UnaryOperator { input: lhs }).into(),
                    "isinf" => Comparison::IsInf(UnaryOperator { input: lhs }).into(),
                    other => return Err(format!("unknown comparison {other}")),
                };
                push(scope, operation, out);
            }
            Stmt::Cast { out, a } => {
                let out = self.local(*out)?;
                push(scope, Operator::Cast(UnaryOperator { input: self.operand(a)? }).into(), out);
            }
            Stmt::Bitcast { out, a } => {
                let out = self.local(*out)?;
                push(scope, Operator::Reinterpret(UnaryOperator { input: self.operand(a)? }).into(), out);
            }
            Stmt::Sel { out, c, a, b } => {
                let out = self.local(*out)?;
                let select = Select { cond: self.operand(c)?, then: self.operand(a)?, or_else: self.operand(b)? };
                push(scope, Operator::Select(select).into(), out);
            }
            Stmt::Mov { out, a } => {
                let out = self.local(*out)?;
                push(scope, Operation::Copy(self.operand(a)?), out);
            }
            Stmt::Ld { out, ptr } => {
                let out = self.local(*out)?;
                push(scope, Operator::NativeLoad(UnaryOperator { input: self.operand(ptr)? }).into(), out);
            }
            Stmt::St { ptr, val } => {
                let store = BinaryOperator { lhs: self.operand(ptr)?, rhs: self.operand(val)? };
                scope.instructions.push(Instruction::no_out(Operator::NativeStore(store)));
            }
            Stmt::Shld { out, sh, idx } => {
                let out = self.local(*out)?;
                let list = *self.shared.get(*sh).ok_or_else(|| format!("unknown shared array {sh}"))?;
                let index = IndexOperator { list, index: self.operand(idx)?, vector_size: 0, unroll_factor: 1 };
                push(scope, Operator::UncheckedIndex(index).into(), out);
            }
            Stmt::Shst { sh, idx, val } => {
                let array = *self.shared.get(*sh).ok_or_else(|| format!("unknown shared array {sh}"))?;
                let assign = IndexAssignOperator {
                    index: self.operand(idx)?,
                    value: self.operand(val)?,
                    vector_size: 0,
                    unroll_factor: 1,
                };
                push(scope, Operator::UncheckedIndexAssign(assign).into(), array);
            }
            Stmt::Bar => scope.instructions.push(Instruction::no_out(Synchronization::SyncRuda)),
            Stmt::If { c, then, otherwise } => {
                let cond = self.operand(c)?;
                let scope_if = self.block(scope, then)?;
                let branch = if otherwise.is_empty() {
                    Branch::If(Box::new(If { cond, scope: scope_if }))
                } else {
                    let scope_else = self.block(scope, otherwise)?;
                    Branch::IfElse(Box::new(IfElse { cond, scope_if, scope_else }))
                };
                scope.instructions.push(Instruction::no_out(branch));
            }
            Stmt::Loop { body } => {
                let inner = self.block(scope, body)?;
                scope.instructions.push(Instruction::no_out(Branch::Loop(Box::new(Loop { scope: inner }))));
            }
            Stmt::Brk => scope.instructions.push(Instruction::no_out(Branch::Break)),
        }
        Ok(())
    }
}

const INFO_ALIGN: usize = 8;

pub fn compile(kernel: &Kernel) -> Result<Compiled, String> {
    let mut locals = kernel.locals.iter().map(|t| parse_type(t)).collect::<Result<Vec<_>, _>>()?;

    // One scalar group per storage type, in first-appearance order: the info ABI addresses a
    // scalar by (group, index) and the PTX emitter looks groups up by type.
    let mut groups: Vec<(StorageType, u32)> = Vec::new();
    let mut scalars = Vec::new();
    let mut slots = Vec::new();
    for name in &kernel.scalars {
        let ty = parse_type(name)?;
        let storage = ty.storage_type();
        let group = match groups.iter().position(|(t, _)| *t == storage) {
            Some(g) => g,
            None => {
                groups.push((storage, 0));
                groups.len() - 1
            }
        };
        let id = groups[group].1;
        groups[group].1 += 1;
        scalars.push((ty, id));
        slots.push((group, id as usize, storage.size()));
    }
    let mut group_offsets = Vec::new();
    let mut offset = 0;
    for (ty, count) in &groups {
        group_offsets.push(offset);
        offset += (ty.size() * *count as usize).next_multiple_of(INFO_ALIGN);
    }
    let scalar_offsets = slots.iter().map(|(g, id, size)| group_offsets[*g] + id * size).collect();

    let mut body = Scope::root(false);
    let mut buffers = Vec::new();
    let mut ptr_bases = Vec::new();
    let u64_ty: Type = UIntKind::U64.into();
    let u8_ty: Type = UIntKind::U8.into();
    for id in 0..kernel.num_ptrs {
        buffers.push(KernelArg { id, visibility: Visibility::ReadWrite, ty: u8_ty, size: None, has_extended_meta: false });
        let local_id = locals.len() as u32;
        locals.push(u64_ty);
        let base = Variable::new(VariableKind::LocalMut { id: local_id }, u64_ty);
        let array = Variable::new(VariableKind::GlobalOutputArray(id), u8_ty);
        let zero = Variable::constant(ConstantValue::UInt(0), UIntKind::U32);
        body.instructions.push(Instruction::new(
            Operator::NativeAddress(BinaryOperator { lhs: array, rhs: zero }),
            base,
        ));
        ptr_bases.push(base);
    }
    let mut shared = Vec::new();
    for array in &kernel.shared {
        shared.push(*body.create_shared_array(parse_type(&array.ty)?, array.len, Some(16)));
    }

    let builder = Builder { locals, scalars, ptr_bases, shared };
    for stmt in &kernel.body {
        builder.stmt(&mut body, stmt)?;
    }

    let definition = KernelDefinition {
        buffers,
        tensor_maps: vec![],
        scalars: groups.iter().map(|(ty, count)| ScalarKernelArg { ty: *ty, count: *count as usize }).collect(),
        ruda_dim: RudaDim::new_1d(kernel.threads),
        body,
        options: KernelOptions { kernel_name: kernel.name.clone(), ..Default::default() },
    };
    let options = PtxCompilationOptions { target: Some(PtxTarget { version: kernel.ptx_version, sm: kernel.sm }) };
    let ptx = PtxCompiler
        .compile(definition, &options, ExecutionMode::Unchecked, UIntKind::U32.into())
        .map_err(|e| format!("{e}"))?;
    let info_size = ptx
        .source
        .split(".b8 info[")
        .nth(1)
        .and_then(|rest| rest.split(']').next())
        .map(|n| n.parse::<usize>().map_err(|e| e.to_string()))
        .transpose()?
        .unwrap_or(0);
    Ok(Compiled {
        name: ptx.entrypoint,
        ptx: ptx.source,
        threads: kernel.threads,
        shared: ptx.shared_memory_bytes,
        info_size,
        scalar_offsets,
    })
}
