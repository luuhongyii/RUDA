use std::collections::{HashMap, HashSet};

use ruda::runtime::config::compilation::F16Evaluation;
use ruda_core::ir::*;

/// Operates before SSA so mutable assignments remain explicit rounding boundaries.
pub(super) fn evaluate(scope: &mut Scope, mode: F16Evaluation) {
    if mode == F16Evaluation::PerOperation {
        return;
    }
    normalize(scope, &mut HashMap::new());
    expand_composites(scope);
    scope.locals.extend(scope.allocator.take_variables());
    let mut candidates = HashMap::new();
    let mut escaped = HashSet::new();
    walk(scope, 0, &mut |scope, depth| {
        for &var in &scope.locals {
            if local_f16(var) {
                candidates.entry(var).or_insert(depth);
            }
        }
        escaped.extend(scope.parameters.iter().copied());
        for inst in &mut scope.instructions {
            if let Some(out) = inst.out.filter(|out| local_f16(*out)) {
                candidates.entry(out).or_insert(depth);
                if !writes_value(&inst.operation) {
                    escaped.insert(out);
                }
            }
            operands(&mut inst.operation, &mut |var, address| {
                if address {
                    escaped.insert(*var);
                }
            });
        }
    });
    candidates.retain(|var, _| !escaped.contains(var));

    normalize(scope, &mut HashMap::new());
    let mut wide = HashMap::new();
    let mut pending = HashMap::new();
    promote_arithmetic(scope, &mut wide, &mut pending);
    insert_casts(scope, &mut pending);
    // Parameters, constants and global scalars have no defining instruction.
    let mut entry: Vec<_> = pending.into_iter().collect();
    entry.sort_by_key(|(_, out)| local_id(*out));
    let mut prefix = Vec::new();
    for (input, out) in entry {
        prefix.push(Instruction::new(Operator::Cast(UnaryOperator { input }), out));
    }
    prefix.append(&mut scope.instructions);
    scope.instructions = prefix;

    if mode == F16Evaluation::Accumulators && !wide.is_empty() {
        promote_accumulators(scope, &candidates, &mut wide);
    }
    // A callback has its own parameter/storage boundary and precision state.
    walk(scope, 0, &mut |scope, _| {
        for inst in &mut scope.instructions {
            if let Operation::CoopMma(CoopMma::Elementwise { function, .. }) = &mut inst.operation {
                evaluate(&mut function.body, mode);
            }
        }
    });
}

fn f16(ty: Type) -> bool {
    matches!(ty, Type::Scalar(StorageType::Scalar(ElemType::Float(FloatKind::F16)))
        | Type::Vector(StorageType::Scalar(ElemType::Float(FloatKind::F16)), _))
}

fn local_f16(var: Variable) -> bool {
    matches!(var.kind, VariableKind::LocalMut { .. }) && f16(var.ty)
}

fn local_id(var: Variable) -> u32 {
    match var.kind {
        VariableKind::LocalMut { id } | VariableKind::LocalConst { id }
        | VariableKind::Versioned { id, .. } => id,
        _ => unreachable!("precision temporary must be local"),
    }
}

fn wide_type(ty: Type) -> Type {
    ty.with_storage_type(StorageType::Scalar(ElemType::Float(FloatKind::F32)))
}

fn derived(inst: &Instruction, operation: impl Into<Operation>, out: Variable) -> Instruction {
    Instruction {
        operation: operation.into(),
        out: Some(out),
        source_loc: inst.source_loc.clone(),
        modes: inst.modes,
    }
}

fn children(scope: &mut Scope, depth: usize, visit: &mut dyn FnMut(&mut Scope, usize)) {
    for inst in &mut scope.instructions {
        if let Operation::Branch(branch) = &mut inst.operation {
            match branch {
                Branch::If(b) => visit(&mut b.scope, depth),
                Branch::IfElse(b) => {
                    visit(&mut b.scope_if, depth);
                    visit(&mut b.scope_else, depth);
                }
                Branch::Switch(b) => {
                    visit(&mut b.scope_default, depth);
                    for (_, scope) in &mut b.cases {
                        visit(scope, depth);
                    }
                }
                Branch::RangeLoop(b) => visit(&mut b.scope, depth + 1),
                Branch::Loop(b) => visit(&mut b.scope, depth + 1),
                Branch::Return | Branch::ReturnValue(_) | Branch::Break | Branch::Unreachable => {}
            }
        }
    }
}

fn walk(scope: &mut Scope, depth: usize, visit: &mut dyn FnMut(&mut Scope, usize)) {
    visit(scope, depth);
    children(scope, depth, &mut |scope, depth| walk(scope, depth, visit));
}

fn writes_value(op: &Operation) -> bool {
    !matches!(op,
        Operation::Operator(Operator::IndexAssign(_) | Operator::UncheckedIndexAssign(_)
            | Operator::CopyMemory(_) | Operator::CopyMemoryBulk(_) | Operator::NativeStore(_))
        | Operation::TypedMemory(TypedMemory::Store { .. })
        | Operation::Pointer(PointerOp::Store { .. })
        | Operation::Atomic(AtomicOp::Store(_))
        | Operation::CoopMma(CoopMma::Store { .. } | CoopMma::StoreTensor { .. }
            | CoopMma::StoreMatrix { .. })
        | Operation::Barrier(_) | Operation::Tma(_)
    )
}

fn normalize(scope: &mut Scope, aliases: &mut HashMap<Variable, Variable>) {
    let mut instructions = Vec::new();
    for mut inst in core::mem::take(&mut scope.instructions) {
        let template = inst.clone();
        operands(&mut inst.operation, &mut |var, address| {
            if address {
                return;
            }
            if let Some(alias) = aliases.get(var) {
                *var = *alias;
            }
            if f16(var.ty) && matches!(var.kind,
                VariableKind::LocalMut { .. } | VariableKind::Shared { .. })
            {
                let loaded = *scope.allocator.create_local(var.ty);
                instructions.push(derived(&template, Operation::Copy(*var), loaded));
                *var = loaded;
            }
        });
        let destination = inst.out.filter(|out| local_f16(*out) && writes_value(&inst.operation));
        if let Some(out) = destination {
            inst.out = Some(*scope.allocator.create_local(out.ty));
        }
        let mut alias = None;
        if let (Operation::Copy(input), Some(out)) = (&inst.operation, inst.out) {
            if matches!(out.kind, VariableKind::LocalConst { .. } | VariableKind::Versioned { .. })
                && input.ty == out.ty && f16(out.ty)
            {
                aliases.insert(out, *input);
                alias = Some(*input);
            }
        }
        let result = alias.or(inst.out);
        if alias.is_none() {
            instructions.push(inst);
        }
        if let Some(out) = destination {
            instructions.push(derived(&template, Operation::Copy(result.unwrap()), out));
        }
    }
    scope.instructions = instructions;
    children(scope, 0, &mut |scope, _| normalize(scope, aliases));
}

fn promotable(op: &Operation) -> bool {
    matches!(op, Operation::Arithmetic(
        Arithmetic::Add(_) | Arithmetic::Sub(_) | Arithmetic::Mul(_) | Arithmetic::Div(_)
        | Arithmetic::Modulo(_) | Arithmetic::Neg(_) | Arithmetic::Fma(_)
        | Arithmetic::Min(_) | Arithmetic::Max(_) | Arithmetic::Sqrt(_)
        | Arithmetic::InverseSqrt(_) | Arithmetic::Recip(_)
    ))
}

fn expand_composites(scope: &mut Scope) {
    use ruda_kernel::dsl::frontend::{expand_erf, expand_hypot, expand_rhypot};
    let mut instructions = Vec::new();
    for inst in core::mem::take(&mut scope.instructions) {
        if !inst.out.is_some_and(|out| f16(out.ty)) {
            instructions.push(inst);
            continue;
        }
        let out = inst.out();
        let mut expanded = scope.child();
        let mut emit = |op: Arithmetic, ty: Type| {
            let value = *expanded.allocator.create_local(ty);
            expanded.instructions.push(derived(&inst, op, value));
            value
        };
        let value = match &inst.operation {
            Operation::Arithmetic(Arithmetic::Remainder(op)) => {
                let quotient = emit(Arithmetic::Div(op.clone()), out.ty);
                let floor = emit(Arithmetic::Floor(UnaryOperator { input: quotient }), out.ty);
                let product = emit(Arithmetic::Mul(BinaryOperator { lhs: floor, rhs: op.rhs }), out.ty);
                Some(emit(Arithmetic::Sub(BinaryOperator { lhs: op.lhs, rhs: product }), out.ty))
            }
            Operation::Arithmetic(Arithmetic::Log1p(op)) => {
                let one = Variable::constant(ConstantValue::Float(1.0), op.input.ty);
                let sum = emit(Arithmetic::Add(BinaryOperator { lhs: op.input, rhs: one }), out.ty);
                Some(emit(Arithmetic::Log(UnaryOperator { input: sum }), out.ty))
            }
            Operation::Arithmetic(Arithmetic::Degrees(op)) => {
                let rhs = Variable::constant(ConstantValue::Float(57.29577951308232), op.input.ty);
                Some(emit(Arithmetic::Mul(BinaryOperator { lhs: op.input, rhs }), out.ty))
            }
            Operation::Arithmetic(Arithmetic::Radians(op)) => {
                let rhs = Variable::constant(ConstantValue::Float(0.017453292519943295), op.input.ty);
                Some(emit(Arithmetic::Mul(BinaryOperator { lhs: op.input, rhs }), out.ty))
            }
            Operation::Arithmetic(Arithmetic::Dot(op)) => {
                let product = emit(Arithmetic::Mul(op.clone()), op.lhs.ty);
                Some(if op.lhs.ty.vector_size() > 1 {
                    emit(Arithmetic::VectorSum(UnaryOperator { input: product }), out.ty)
                } else { product })
            }
            Operation::Arithmetic(Arithmetic::Magnitude(op)) => {
                let squared = emit(Arithmetic::Dot(BinaryOperator {
                    lhs: op.input, rhs: op.input,
                }), op.input.ty.with_vector_size(1));
                Some(emit(Arithmetic::Sqrt(UnaryOperator { input: squared }), out.ty))
            }
            Operation::Arithmetic(Arithmetic::Normalize(op)) => {
                let scalar = op.input.ty.with_vector_size(1);
                let denominator = if op.input.ty.vector_size() > 1 {
                    emit(Arithmetic::Magnitude(op.clone()), scalar)
                } else {
                    emit(Arithmetic::Abs(op.clone()), scalar)
                };
                Some(emit(Arithmetic::Div(BinaryOperator {
                    lhs: op.input, rhs: denominator,
                }), out.ty))
            }
            Operation::Arithmetic(Arithmetic::Clamp(op)) => {
                drop(emit);
                let mut splat = |input: Variable| {
                    if input.ty.vector_size() == out.ty.vector_size() {
                        input
                    } else {
                        let value = *expanded.allocator.create_local(out.ty);
                        expanded.instructions.push(derived(&inst, Operator::InitVector(
                            VectorInitOperator { inputs: vec![input; out.ty.vector_size()] }
                        ), value));
                        value
                    }
                };
                let min = splat(op.min_value);
                let max = splat(op.max_value);
                let lower = *expanded.allocator.create_local(out.ty);
                expanded.instructions.push(derived(&inst, Arithmetic::Max(BinaryOperator {
                    lhs: op.input, rhs: min,
                }), lower));
                expanded.instructions.push(derived(&inst, Arithmetic::Min(BinaryOperator {
                    lhs: lower, rhs: max,
                }), out));
                None
            }
            Operation::Arithmetic(Arithmetic::Hypot(op)) => {
                drop(emit);
                expand_hypot(&mut expanded, op.lhs, op.rhs, out);
                None
            }
            Operation::Arithmetic(Arithmetic::Rhypot(op)) => {
                drop(emit);
                expand_rhypot(&mut expanded, op.lhs, op.rhs, out);
                None
            }
            Operation::Arithmetic(Arithmetic::Erf(op)) => {
                drop(emit);
                expand_erf(&mut expanded, op.input, out);
                None
            }
            Operation::Arithmetic(Arithmetic::VectorSum(op)) if op.input.ty.vector_size() == 1 => {
                Some(op.input)
            }
            _ => {
                instructions.push(inst);
                continue;
            }
        };
        if let Some(value) = value {
            expanded.instructions.push(derived(&inst, Operation::Copy(value), out));
        }
        expand_composites(&mut expanded);
        scope.locals.append(&mut expanded.locals);
        instructions.append(&mut expanded.instructions);
    }
    scope.instructions = instructions;
    children(scope, 0, &mut |scope, _| expand_composites(scope));
}

fn promote_arithmetic(
    scope: &mut Scope,
    wide: &mut HashMap<Variable, Variable>,
    pending: &mut HashMap<Variable, Variable>,
) {
    let mut instructions = Vec::new();
    for mut inst in core::mem::take(&mut scope.instructions) {
        if promotable(&inst.operation) && inst.out.is_some_and(|out| f16(out.ty)) {
            let original = inst.clone();
            let out = inst.out();
            operands(&mut inst.operation, &mut |input, _| {
                if input.ty.vector_size() == 1 && out.ty.vector_size() > 1 {
                    let broadcast = *scope.allocator.create_local(out.ty);
                    instructions.push(derived(&original, Operator::InitVector(VectorInitOperator {
                        inputs: vec![*input; out.ty.vector_size()],
                    }), broadcast));
                    *input = broadcast;
                }
                *input = *wide.entry(*input).or_insert_with(|| {
                    let value = *scope.allocator.create_local(wide_type(input.ty));
                    pending.insert(*input, value);
                    value
                });
            });
            let result = *scope.allocator.create_local(wide_type(out.ty));
            inst.out = Some(result);
            instructions.push(inst);
            instructions.push(derived(&original, Operator::Cast(UnaryOperator { input: result }), out));
            wide.insert(out, result);
        } else {
            instructions.push(inst);
        }
    }
    scope.instructions = instructions;
    children(scope, 0, &mut |scope, _| promote_arithmetic(scope, wide, pending));
}

fn insert_casts(scope: &mut Scope, pending: &mut HashMap<Variable, Variable>) {
    let mut instructions = Vec::new();
    for inst in core::mem::take(&mut scope.instructions) {
        let cast = inst.out.and_then(|input| pending.remove(&input).map(|out| {
            derived(&inst, Operator::Cast(UnaryOperator { input }), out)
        }));
        instructions.push(inst);
        instructions.extend(cast);
    }
    scope.instructions = instructions;
    children(scope, 0, &mut |scope, _| insert_casts(scope, pending));
}

#[derive(Default)]
struct Accesses {
    loads: Vec<Variable>,
    stores: Vec<(Variable, usize)>,
}

#[derive(Clone, Copy)]
struct Use {
    widening: bool,
    stored_into: Option<usize>,
    depth: usize,
}

#[derive(Default)]
struct Converts {
    removed: usize,
    added: usize,
}

impl Converts {
    fn record(&mut self, removed: bool) {
        if removed { self.removed += 1; } else { self.added += 1; }
    }

    fn verdict(&self) -> Option<bool> {
        (self.removed != self.added).then_some(self.removed > self.added)
    }
}

fn root(parents: &[usize], mut index: usize) -> usize {
    while parents[index] != index {
        index = parents[index];
    }
    index
}

fn widening(inst: &Instruction) -> bool {
    matches!(&inst.operation, Operation::Operator(Operator::Cast(op))
        if f16(op.input.ty) && inst.out.is_some_and(|out| out.ty == wide_type(op.input.ty)))
}

fn promote_accumulators(
    scope: &mut Scope,
    declarations: &HashMap<Variable, usize>,
    wide: &mut HashMap<Variable, Variable>,
) {
    let mut variables: Vec<_> = declarations.keys().copied().collect();
    variables.sort_by_key(|var| local_id(*var));
    let owner: HashMap<_, _> = variables.iter().enumerate().map(|(i, &var)| (var, i)).collect();
    let mut accesses: Vec<Accesses> = variables.iter().map(|_| Accesses::default()).collect();
    let mut loaded_from = HashMap::new();
    let mut uses: HashMap<Variable, Vec<Use>> = HashMap::new();
    let mut load_depth = HashMap::new();
    walk(scope, 0, &mut |scope, depth| {
        for inst in &mut scope.instructions {
            if let (Operation::Copy(input), Some(out)) = (&inst.operation, inst.out) {
                if let Some(&index) = owner.get(input) {
                    accesses[index].loads.push(out);
                    loaded_from.insert(out, index);
                    load_depth.insert(out, depth);
                }
                if let Some(&index) = owner.get(&out) {
                    accesses[index].stores.push((*input, depth));
                }
            }
            let usage = Use {
                widening: widening(inst),
                stored_into: if matches!(inst.operation, Operation::Copy(_)) {
                    inst.out.and_then(|out| owner.get(&out).copied())
                } else { None },
                depth,
            };
            operands(&mut inst.operation, &mut |input, address| {
                if !address && f16(input.ty) {
                    uses.entry(*input).or_default().push(usage);
                }
            });
        }
    });
    let mut parents: Vec<_> = (0..variables.len()).collect();
    for (index, access) in accesses.iter().enumerate() {
        for (input, _) in &access.stores {
            if let Some(&source) = loaded_from.get(input) {
                let a = root(&parents, index);
                let b = root(&parents, source);
                parents[b] = a;
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut group_of = HashMap::new();
    for index in 0..variables.len() {
        let root = root(&parents, index);
        let group = *group_of.entry(root).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[group].push(index);
    }
    let mut held = HashMap::new();
    let mut promoted_loads = HashMap::new();
    for group in groups {
        let inside: HashSet<_> = group.iter().copied().collect();
        let declared_at = group.iter().map(|&i| declarations[&variables[i]]).min().unwrap();
        let mut per_iteration = Converts::default();
        let mut once = Converts::default();
        let mut ledger = |depth: usize, removed: bool| {
            if depth > declared_at { &mut per_iteration } else { &mut once }.record(removed);
        };
        for &index in &group {
            for &load in &accesses[index].loads {
                let mut narrowed = false;
                for user in uses.get(&load).into_iter().flatten() {
                    if user.stored_into.is_some_and(|i| inside.contains(&i)) {
                        continue;
                    }
                    if user.widening {
                        ledger(user.depth, true);
                    } else {
                        narrowed = true;
                    }
                }
                if narrowed {
                    ledger(load_depth[&load], false);
                }
            }
            for (input, depth) in &accesses[index].stores {
                if loaded_from.get(input).is_some_and(|i| inside.contains(i)) {
                    continue;
                }
                ledger(*depth, wide.contains_key(input));
            }
        }
        if per_iteration.verdict().or_else(|| once.verdict()) != Some(true) {
            continue;
        }
        for index in group {
            let variable = variables[index];
            let storage = *scope.allocator.create_local_restricted(wide_type(variable.ty));
            scope.locals.push(storage);
            held.insert(variable, storage);
            for &load in &accesses[index].loads {
                let value = *scope.allocator.create_local(wide_type(load.ty));
                promoted_loads.insert(load, value);
                wide.insert(load, value);
            }
        }
    }
    walk(scope, 0, &mut |scope, _| {
        let mut instructions = Vec::new();
        for mut inst in core::mem::take(&mut scope.instructions) {
            let original = inst.clone();
            if let (Operation::Copy(input), Some(out)) = (&inst.operation, inst.out) {
                if let Some(&storage) = held.get(input) {
                    let value = promoted_loads[&out];
                    instructions.push(derived(&inst, Operation::Copy(storage), value));
                    instructions.push(derived(&inst, Operator::Cast(UnaryOperator { input: value }), out));
                    continue;
                }
                if let Some(&storage) = held.get(&out) {
                    let value = if let Some(&value) = wide.get(input) {
                        value
                    } else {
                        let value = *scope.allocator.create_local(wide_type(input.ty));
                        instructions.push(derived(&inst, Operator::Cast(UnaryOperator { input: *input }), value));
                        value
                    };
                    inst = derived(&inst, Operation::Copy(value), storage);
                }
            } else if widening(&inst) {
                if let Operation::Operator(Operator::Cast(op)) = &inst.operation {
                    if let Some(&value) = promoted_loads.get(&op.input) {
                        inst = derived(&original, Operation::Copy(value), inst.out());
                    }
                }
            }
            instructions.push(inst);
        }
        scope.instructions = instructions;
    });
}

/// Enumerate reads without converting references to storage into loaded values.
fn operands(op: &mut Operation, visit: &mut dyn FnMut(&mut Variable, bool)) {
    match op {
        Operation::Branch(branch) => match branch {
            Branch::If(b) => visit(&mut b.cond, false),
            Branch::IfElse(b) => visit(&mut b.cond, false),
            Branch::Switch(b) => visit(&mut b.value, false),
            Branch::RangeLoop(b) => {
                visit(&mut b.start, false);
                visit(&mut b.end, false);
                if let Some(step) = &mut b.step { visit(step, false); }
            }
            Branch::ReturnValue(value) => visit(value, false),
            Branch::Loop(_) | Branch::Return | Branch::Break | Branch::Unreachable => {}
        },
        Operation::NonSemantic(NonSemantic::Print { args, .. }) => {
            for arg in args { visit(arg, false); }
        }
        Operation::NonSemantic(_) | Operation::Marker(_) => {}
        Operation::CoopMma(cmma) => matrix_operands(cmma, visit),
        _ => {
            let code = op.op_code();
            if let Some(mut args) = op.args() {
                for (index, arg) in args.iter_mut().enumerate() {
                    let address = match &*op {
                        Operation::Operator(Operator::Index(_) | Operator::UncheckedIndex(_)
                            | Operator::WorkgroupUniformLoad(_)
                            | Operator::NativeAddress(_)) => index == 0,
                        Operation::Operator(Operator::CopyMemory(_) | Operator::CopyMemoryBulk(_)) => index == 1,
                        Operation::TypedMemory(_) => index == 0,
                        Operation::Pointer(PointerOp::Address(_)) => index == 0,
                        Operation::Metadata(_) => true,
                        Operation::Barrier(_) | Operation::Tma(_) => true,
                        Operation::Atomic(AtomicOp::Store(_)) => false,
                        Operation::Atomic(_) => index == 0,
                        _ => false,
                    };
                    visit(arg, address);
                }
                *op = Operation::from_code_and_args(code, &args).expect("reflected operation");
            }
        }
    }
}

fn matrix_operands(op: &mut CoopMma, visit: &mut dyn FnMut(&mut Variable, bool)) {
    match op {
        CoopMma::Fill { value } => visit(value, false),
        CoopMma::Cast { input } => visit(input, true),
        CoopMma::Load { value, stride, offset, .. } => {
            visit(value, true); visit(stride, false); visit(offset, false);
        }
        CoopMma::Execute { mat_a, mat_b, mat_c } => {
            visit(mat_a, true); visit(mat_b, true); visit(mat_c, true);
        }
        CoopMma::Store { mat, stride, offset, .. } => {
            visit(mat, true); visit(stride, false); visit(offset, false);
        }
        CoopMma::Elementwise { input, captures, .. } => {
            visit(input, true);
            for capture in captures { visit(capture, false); }
        }
        CoopMma::CreateTensorLayout { shape, strides } => {
            for value in shape { visit(value, false); }
            if let Some(strides) = strides { for value in strides { visit(value, false); } }
        }
        CoopMma::CreateTensorView => {}
        CoopMma::SliceTensorLayout { layout, offsets, shape } => {
            visit(layout, true);
            for value in offsets { visit(value, false); }
            for value in shape { visit(value, false); }
        }
        CoopMma::LoadTensor { buffer, offset, layout, view } => {
            visit(buffer, true); visit(offset, false); visit(layout, true);
            if let Some(view) = view { visit(view, true); }
        }
        CoopMma::StoreTensor { mat, offset, layout, view } => {
            visit(mat, true); visit(offset, false); visit(layout, true);
            if let Some(view) = view { visit(view, true); }
        }
        CoopMma::RowIndex { lane_id, i, .. } | CoopMma::ColIndex { lane_id, i, .. } => {
            visit(lane_id, false); visit(i, false);
        }
        CoopMma::LoadMatrix { buffer, offset, .. } => {
            visit(buffer, true); visit(offset, false);
        }
        CoopMma::StoreMatrix { offset, registers, .. } => {
            visit(offset, false); visit(registers, false);
        }
        CoopMma::ExecuteManual { registers_a, registers_b, registers_c, .. } => {
            visit(registers_a, false); visit(registers_b, false); visit(registers_c, false);
        }
        CoopMma::ExecuteScaled { registers_a, registers_b, registers_c, scales_a, scales_b, .. } => {
            visit(registers_a, false); visit(registers_b, false); visit(registers_c, false);
            visit(scales_a, false); visit(scales_b, false);
        }
    }
}
