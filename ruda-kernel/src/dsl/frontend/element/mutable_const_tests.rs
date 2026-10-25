use crate::dsl::prelude::*;
use ruda_core::ir::{Branch, ConstantValue, Operation, Variable, VariableKind};

#[ruda]
fn split_bound(#[comptime] k: usize, #[comptime] bk: usize, last: bool) -> usize {
    let mut blocks = k / bk;
    if last { blocks = (k - bk) / bk; }
    let mut count = 0usize;
    for _ in 0usize..blocks { count += 1; }
    count
}

#[ruda]
fn parameter_bound(#[comptime] initial: u32, condition: bool) -> u32 {
    let mut value = initial;
    if condition { value = 7u32; }
    value
}

#[ruda]
fn float_bound(#[comptime] initial: f32, condition: bool) -> f32 {
    let mut value = initial * 2.0;
    if condition { value = -3.0f32; }
    value
}

#[ruda]
fn boolean_bound(#[comptime] initial: bool, condition: bool) -> bool {
    let mut value = initial;
    if condition { value = false; }
    value
}

fn conditional_assignment(scope: &Scope, initial: ConstantValue, changed: ConstantValue) -> Variable {
    let target = scope.instructions.iter().find_map(|inst| match inst.operation {
        Operation::Copy(Variable { kind: VariableKind::Constant(value), .. }) if value == initial => inst.out,
        _ => None,
    }).expect("mutable constant initializer must be emitted to IR");
    assert!(matches!(target.kind, VariableKind::LocalMut { .. }));
    let arm = scope.instructions.iter().find_map(|inst| match &inst.operation {
        Operation::Branch(Branch::If(branch)) => Some(&branch.scope),
        _ => None,
    }).expect("runtime branch must be present");
    assert!(arm.instructions.iter().any(|inst| inst.out == Some(target)
        && matches!(inst.operation, Operation::Copy(Variable {
            kind: VariableKind::Constant(value), ..
        }) if value == changed)), "branch must update the same runtime binding");
    target
}

#[test]
fn mutable_const_expression_is_a_runtime_loop_bound() {
    let mut scope = Scope::root(false);
    let condition = scope.create_local(ElemType::Bool.into()).into();
    split_bound::__expand(&mut scope, 1216, 64, condition);
    let target = conditional_assignment(&scope, ConstantValue::UInt(19), ConstantValue::UInt(18));
    let bound = scope.instructions.iter().find_map(|inst| match &inst.operation {
        Operation::Branch(Branch::RangeLoop(branch)) => Some(branch.end),
        _ => None,
    }).expect("runtime loop must be present");
    assert_eq!(bound, target);
}

#[test]
fn mutable_const_parameters_and_folded_floats_keep_branch_updates() {
    let mut scope = Scope::root(false);
    let condition = scope.create_local(ElemType::Bool.into()).into();
    parameter_bound::__expand(&mut scope, 11, condition);
    conditional_assignment(&scope, ConstantValue::UInt(11), ConstantValue::UInt(7));

    let mut scope = Scope::root(false);
    let condition = scope.create_local(ElemType::Bool.into()).into();
    float_bound::__expand(&mut scope, 2.0, condition);
    conditional_assignment(&scope, ConstantValue::Float(4.0), ConstantValue::Float(-3.0));

    let mut scope = Scope::root(false);
    let condition = scope.create_local(ElemType::Bool.into()).into();
    boolean_bound::__expand(&mut scope, true, condition);
    conditional_assignment(&scope, ConstantValue::Bool(true), ConstantValue::Bool(false));
}

#[test]
fn mutable_const_nonprimitive_fallback_preserves_host_values() {
    use super::{PrimitiveRuntime, PrimitiveRuntimeFallback};
    let mut scope = Scope::root(false);
    let values = alloc::vec![3usize, 5];
    assert_eq!(PrimitiveRuntime(values.clone()).primitive_runtime(&mut scope), values);
    assert!(scope.instructions.is_empty());
}
