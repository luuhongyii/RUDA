use ruda_core::ir::{
    Arithmetic, Bitwise, Comparison, ConstantValue, Instruction, Metadata, Operation, Operator,
    Type, Variable, VariableKind,
};

use crate::optimizer::{AtomicCounter, Optimizer};

use super::OptimizerPass;

/// Simplifies certain expressions where one operand is constant.
/// For example: `out = x * 1` to `out = x`
pub struct ConstOperandSimplify;

impl OptimizerPass for ConstOperandSimplify {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        for node in opt.program.node_indices().collect::<Vec<_>>() {
            let ops = opt.program[node].ops.borrow().indices().collect::<Vec<_>>();

            for idx in ops {
                simplify_const_operand(&mut opt.program[node].ops.borrow_mut()[idx], &changes);
            }
        }
    }
}

fn simplify_const_operand(op: &mut Instruction, changes: &AtomicCounter) {
    // Rules that absorb or drop an operand only hold for integers: with floats they
    // would change the result for NaN, infinities, signed zeros or fractional values.
    let is_int = op.out.is_some_and(|out| out.ty.is_int());
    match &mut op.operation {
        Operation::Arithmetic(operator) => match operator {
            // 0 * x == 0
            Arithmetic::Mul(bin_op) if is_int && bin_op.lhs.is_constant(0) => {
                op.operation = Operation::Copy(bin_op.lhs);
                changes.inc();
            }
            // x * 0 == 0
            Arithmetic::Mul(bin_op) if is_int && bin_op.rhs.is_constant(0) => {
                op.operation = Operation::Copy(bin_op.rhs);
                changes.inc();
            }
            // 1 * x == x
            Arithmetic::Mul(bin_op) if bin_op.lhs.is_constant(1) => {
                op.operation = Operation::Copy(bin_op.rhs);
                changes.inc();
            }
            // x * 1 == x
            Arithmetic::Mul(bin_op) if bin_op.rhs.is_constant(1) => {
                op.operation = Operation::Copy(bin_op.lhs);
                changes.inc();
            }
            // 0 + x = x (for floats only -0.0 is an exact additive identity)
            Arithmetic::Add(bin_op) if is_additive_identity(&bin_op.lhs, is_int) => {
                op.operation = Operation::Copy(bin_op.rhs);
                changes.inc();
            }
            // x + 0 = x
            Arithmetic::Add(bin_op) if is_additive_identity(&bin_op.rhs, is_int) => {
                op.operation = Operation::Copy(bin_op.lhs);
                changes.inc();
            }
            // x - 0 == x (for floats only +0.0 keeps the sign of a zero x)
            Arithmetic::Sub(bin_op) if is_subtractive_identity(&bin_op.rhs, is_int) => {
                op.operation = Operation::Copy(bin_op.lhs);
                changes.inc();
            }
            // x / 1 == x, 0 / x == 0
            Arithmetic::Div(bin_op)
                if (is_int && bin_op.lhs.is_constant(0) && !bin_op.rhs.is_constant(0)) || bin_op.rhs.is_constant(1) =>
            {
                op.operation = Operation::Copy(bin_op.lhs);
                changes.inc();
            }
            // x % 1 == 0, 0 % x == 0
            Arithmetic::Modulo(bin_op)
                if is_int && !bin_op.rhs.is_constant(0) && (bin_op.rhs.is_constant(1) || bin_op.lhs.is_constant(0)) =>
            {
                op.operation = Operation::Copy(op.ty().constant(ConstantValue::Int(0)));
                changes.inc();
            }
            _ => {}
        },

        Operation::Operator(operator) => match operator {
            // true || x == true, x || true == true
            Operator::Or(bin_op) if bin_op.lhs.is_true() || bin_op.rhs.is_true() => {
                op.operation = Operation::Copy(true.into());
                changes.inc();
            }
            // false || x == x, x || false == x
            Operator::Or(bin_op) if bin_op.lhs.is_false() => {
                op.operation = Operation::Copy(bin_op.rhs);
                changes.inc();
            }
            // x || false == x
            Operator::Or(bin_op) if bin_op.rhs.is_false() => {
                op.operation = Operation::Copy(bin_op.lhs);
                changes.inc();
            }
            // false && x == false, x && false == false
            Operator::And(bin_op) if bin_op.lhs.is_false() || bin_op.rhs.is_false() => {
                op.operation = Operation::Copy(false.into());
                changes.inc();
            }
            // true && x == x
            Operator::And(bin_op) if bin_op.lhs.is_true() => {
                op.operation = Operation::Copy(bin_op.rhs);
                changes.inc();
            }
            // x && true == x
            Operator::And(bin_op) if bin_op.rhs.is_true() => {
                op.operation = Operation::Copy(bin_op.lhs);
                changes.inc();
            }
            // select(true, a, b) == a
            Operator::Select(select) if select.cond.is_true() => {
                op.operation = Operation::Copy(select.then);
                changes.inc();
            }
            // select(false, a, b) == b
            Operator::Select(select) if select.cond.is_false() => {
                op.operation = Operation::Copy(select.or_else);
                changes.inc();
            }
            _ => {}
        },

        Operation::Pointer(ruda_core::ir::PointerOp::Select(select))
            if select.cond.is_true()
                || select.cond.is_false()
                || select.then == select.or_else =>
        {
            op.operation = Operation::Copy(if select.cond.is_false() {
                select.or_else
            } else {
                select.then
            });
            changes.inc();
        }
        // Constant length to const value
        Operation::Metadata(Metadata::Length { var }) => match var.kind {
            VariableKind::ConstantArray { length, .. }
            | VariableKind::SharedArray { length, .. }
            | VariableKind::LocalArray { length, .. } => {
                op.operation = Operation::Copy(length.into());
                changes.inc();
            }
            _ => {}
        },
        _ => {}
    }
}

fn float_zero_sign(var: &Variable) -> Option<bool> {
    match var.kind {
        VariableKind::Constant(ConstantValue::Float(val)) if val == 0.0 => {
            Some(val.is_sign_negative())
        }
        _ => None,
    }
}

fn is_additive_identity(var: &Variable, is_int: bool) -> bool {
    if is_int {
        var.is_constant(0)
    } else {
        float_zero_sign(var) == Some(true)
    }
}

fn is_subtractive_identity(var: &Variable, is_int: bool) -> bool {
    if is_int {
        var.is_constant(0)
    } else {
        float_zero_sign(var) == Some(false)
    }
}

/// Evaluates expressions where both operands are constant and replaces them with simple constant
/// assignments. This can often be applied as a result of assignment merging or constant operand
/// simplification.
pub struct ConstEval;

impl OptimizerPass for ConstEval {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        for node in opt.node_ids() {
            let ops = opt.program[node].ops.clone();
            for op in ops.borrow_mut().values_mut() {
                if let Some(const_eval) = try_const_eval(op) {
                    let input = Variable::constant(const_eval, op.out().ty);
                    op.operation = Operation::Copy(input);
                    changes.inc();
                }
            }
        }
    }
}

macro_rules! const_eval {
    ($op:tt $lhs:expr, $rhs:expr) => {{
        use ConstantValue::*;

        let ty = $lhs.ty;
        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let rhs = rhs.cast_to(ty);
            Some(match (lhs, rhs) {
                (Int(lhs), Int(rhs)) => ConstantValue::Int(lhs $op rhs),
                (Float(lhs), Float(rhs)) => ConstantValue::Float(lhs $op rhs),
                (UInt(lhs), UInt(rhs)) => ConstantValue::UInt(lhs $op rhs),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
    ($lhs:expr, $rhs:expr; $op:path) => {{
        use ConstantValue::*;

        let ty = $lhs.ty;
        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let rhs = rhs.cast_to(ty);
            Some(match (lhs, rhs) {
                (Int(lhs), Int(rhs)) => ConstantValue::Int($op(lhs, rhs)),
                (Float(lhs), Float(rhs)) => ConstantValue::Float($op(lhs, rhs)),
                (UInt(lhs), UInt(rhs)) => ConstantValue::UInt($op(lhs, rhs)),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

macro_rules! const_eval_int {
    ($op:tt $lhs:expr, $rhs:expr) => {{
        use ConstantValue::*;

        let ty = $lhs.ty;
        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let rhs = rhs.cast_to(ty);
            Some(match (lhs, rhs) {
                (Int(lhs), Int(rhs)) => ConstantValue::Int(lhs $op rhs),
                (UInt(lhs), UInt(rhs)) => ConstantValue::UInt(lhs $op rhs),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

macro_rules! const_eval_float {
    ($lhs:expr, $rhs:expr; $fn:path) => {{
        use ConstantValue::*;

        let ty = $lhs.ty;
        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let rhs = rhs.cast_to(ty);
            Some(match (lhs, rhs) {
                (Float(lhs), Float(rhs)) => ConstantValue::Float($fn(lhs, rhs)),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
    ($input:expr; $fn:path) => {{
        use ConstantValue::*;

        if let Some(input) = $input.as_const() {
            Some(match input {
                Float(input) => ConstantValue::Float($fn(input)),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

macro_rules! const_eval_cmp {
    ($op:tt $lhs:expr, $rhs:expr) => {{
        use ConstantValue::*;

        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            Some(match (lhs, rhs) {
                (Int(lhs), Int(rhs)) => ConstantValue::Bool(lhs $op rhs),
                (Float(lhs), Float(rhs)) => ConstantValue::Bool(lhs $op rhs),
                (UInt(lhs), UInt(rhs)) => ConstantValue::Bool(lhs $op rhs),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

macro_rules! const_eval_bool {
    ($op:tt $lhs:expr, $rhs:expr) => {{
        use ConstantValue::*;

        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            Some(match (lhs, rhs) {
                (Bool(lhs), Bool(rhs)) => ConstantValue::Bool(lhs $op rhs),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

fn try_const_eval(inst: &mut Instruction) -> Option<ConstantValue> {
    match &mut inst.operation {
        Operation::Arithmetic(op) => try_const_eval_arithmetic(op),
        Operation::Comparison(op) => try_const_eval_cmp(op),
        Operation::Bitwise(op) => try_const_eval_bitwise(op),
        Operation::Operator(op) => try_const_eval_operator(op, inst.out.map(|it| it.ty)),
        _ => None,
    }
}

fn const_integer_power<F: num::Float>(mut base: F, negative: bool, mut exponent: u64) -> F {
    let mut result = F::one();
    if negative {
        base = F::one() / base;
    }
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = result * base;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = base * base;
        }
    }
    result
}

fn has_int_zero_divisor(op: &ruda_core::ir::BinaryOperator) -> bool {
    !op.lhs.ty.is_float() && op.rhs.is_constant(0)
}

/// Floating remainder with the sign of the divisor, matching `torch.remainder`.
fn floor_remainder<F: num::Float>(lhs: F, rhs: F) -> F {
    let rem = lhs % rhs;
    if rem != F::zero() && (rem < F::zero()) != (rhs < F::zero()) {
        rem + rhs
    } else {
        rem
    }
}

fn try_const_eval_arithmetic(op: &mut Arithmetic) -> Option<ConstantValue> {
    match op {
        Arithmetic::Add(op) => const_eval!(+ op.lhs, op.rhs),
        Arithmetic::Sub(op) => const_eval!(-op.lhs, op.rhs),
        Arithmetic::Mul(op) => const_eval!(*op.lhs, op.rhs),
        Arithmetic::Div(op) if has_int_zero_divisor(op) => None,
        Arithmetic::Div(op) => const_eval!(/ op.lhs, op.rhs),
        Arithmetic::SaturatingAdd(op) => {
            use ConstantValue::*;

            let ty = op.lhs.ty;
            let lhs = op.lhs.as_const();
            let rhs = op.rhs.as_const();
            if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
                let rhs = rhs.cast_to(ty);
                // Saturating ops only support 32-bit
                // NOTE: Change this if that's ever not the case!
                Some(match (lhs, rhs) {
                    (Int(lhs), Int(rhs)) => Int((lhs as i32).saturating_add(rhs as i32) as i64),
                    (UInt(lhs), UInt(rhs)) => UInt((lhs as u32).saturating_add(rhs as u32) as u64),
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Arithmetic::SaturatingSub(op) => {
            use ConstantValue::*;

            let ty = op.lhs.ty;
            let lhs = op.lhs.as_const();
            let rhs = op.rhs.as_const();
            if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
                let rhs = rhs.cast_to(ty);
                // Saturating ops only support 32-bit
                // NOTE: Change this if that's ever not the case!
                Some(match (lhs, rhs) {
                    (Int(lhs), Int(rhs)) => Int((lhs as i32).saturating_sub(rhs as i32) as i64),
                    (UInt(lhs), UInt(rhs)) => UInt((lhs as u32).saturating_sub(rhs as u32) as u64),
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Arithmetic::Powf(op) => const_eval_float!(op.lhs, op.rhs; num::Float::powf),
        Arithmetic::Powi(op) => {
            use ruda_core::ir::{ElemType, FloatKind};

            let ConstantValue::Float(base) = op.lhs.as_const()?.cast_to(op.lhs.ty) else {
                return None;
            };
            let (negative, exponent) = match op.rhs.as_const()?.cast_to(op.rhs.ty) {
                ConstantValue::Int(value) => (value < 0, value.unsigned_abs()),
                ConstantValue::UInt(value) => (false, value),
                _ => return None,
            };
            let result = match op.lhs.ty.elem_type() {
                ElemType::Float(FloatKind::F64) => const_integer_power(base, negative, exponent),
                ElemType::Float(FloatKind::F32 | FloatKind::Flex32 | FloatKind::TF32 |
                    FloatKind::F16 | FloatKind::BF16) => {
                    const_integer_power(base as f32, negative, exponent) as f64
                }
                _ => return None,
            };
            Some(ConstantValue::Float(result).cast_to(op.lhs.ty))
        }
        Arithmetic::Modulo(op) | Arithmetic::Remainder(op) if has_int_zero_divisor(op) => None,
        Arithmetic::Modulo(op) => const_eval!(% op.lhs, op.rhs),
        Arithmetic::Remainder(op) if op.lhs.ty.is_float() => {
            let ConstantValue::Float(lhs) = op.lhs.as_const()?.cast_to(op.lhs.ty) else {
                return None;
            };
            let ConstantValue::Float(rhs) = op.rhs.as_const()?.cast_to(op.lhs.ty) else {
                return None;
            };
            let result = match op.lhs.ty.elem_type() {
                ruda_core::ir::ElemType::Float(ruda_core::ir::FloatKind::F64) => {
                    floor_remainder(lhs, rhs)
                }
                _ => floor_remainder(lhs as f32, rhs as f32) as f64,
            };
            Some(ConstantValue::Float(result))
        }
        Arithmetic::Remainder(op) => const_eval!(% op.lhs, op.rhs),
        Arithmetic::MulHi(op) => {
            use ConstantValue::*;
            let ty = op.lhs.ty;
            if let (Some(lhs), Some(rhs)) = (op.lhs.as_const(), op.rhs.as_const()) {
                let rhs = rhs.cast_to(ty);
                Some(match (lhs, rhs) {
                    (Int(lhs), Int(rhs)) => {
                        let mul = (lhs * rhs) >> 32;
                        ConstantValue::Int(mul as i32 as i64)
                    }
                    (UInt(lhs), UInt(rhs)) => {
                        let mul = (lhs * rhs) >> 32;
                        ConstantValue::UInt(mul as u32 as u64)
                    }
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Arithmetic::Max(op) => {
            use ConstantValue::*;
            let ty = op.lhs.ty;
            if let (Some(lhs), Some(rhs)) = (op.lhs.as_const(), op.rhs.as_const()) {
                let rhs = rhs.cast_to(ty);
                Some(match (lhs, rhs) {
                    (Int(lhs), Int(rhs)) => ConstantValue::Int(lhs.max(rhs)),
                    (Float(lhs), Float(rhs)) => ConstantValue::Float(lhs.max(rhs)),
                    (UInt(lhs), UInt(rhs)) => ConstantValue::UInt(lhs.max(rhs)),
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Arithmetic::Min(op) => {
            use ConstantValue::*;
            let ty = op.lhs.ty;
            if let (Some(lhs), Some(rhs)) = (op.lhs.as_const(), op.rhs.as_const()) {
                let rhs = rhs.cast_to(ty);
                Some(match (lhs, rhs) {
                    (Int(lhs), Int(rhs)) => ConstantValue::Int(lhs.min(rhs)),
                    (Float(lhs), Float(rhs)) => ConstantValue::Float(lhs.min(rhs)),
                    (UInt(lhs), UInt(rhs)) => ConstantValue::UInt(lhs.min(rhs)),
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Arithmetic::Dot(op) => const_eval!(*op.lhs, op.rhs),

        Arithmetic::Abs(op) => {
            use ConstantValue::*;
            op.input.as_const().map(|input| match input {
                Int(input) => ConstantValue::Int(input.abs()),
                Float(input) => ConstantValue::Float(input.abs()),
                _ => unreachable!(),
            })
        }
        Arithmetic::Exp(op) => const_eval_float!(op.input; num::Float::exp),
        Arithmetic::Log(op) => const_eval_float!(op.input; num::Float::ln),
        Arithmetic::Log1p(op) => const_eval_float!(op.input; num::Float::ln_1p),
        Arithmetic::Cos(op) => const_eval_float!(op.input; num::Float::cos),
        Arithmetic::Sin(op) => const_eval_float!(op.input; num::Float::sin),
        Arithmetic::Tan(op) => const_eval_float!(op.input; num::Float::tan),
        Arithmetic::Tanh(op) => const_eval_float!(op.input; num::Float::tanh),
        Arithmetic::Sinh(op) => const_eval_float!(op.input; num::Float::sinh),
        Arithmetic::Cosh(op) => const_eval_float!(op.input; num::Float::cosh),
        Arithmetic::ArcCos(op) => const_eval_float!(op.input; num::Float::acos),
        Arithmetic::ArcSin(op) => const_eval_float!(op.input; num::Float::asin),
        Arithmetic::ArcTan(op) => const_eval_float!(op.input; num::Float::atan),
        Arithmetic::ArcSinh(op) => const_eval_float!(op.input; num::Float::asinh),
        Arithmetic::ArcCosh(op) => const_eval_float!(op.input; num::Float::acosh),
        Arithmetic::ArcTanh(op) => const_eval_float!(op.input; num::Float::atanh),
        Arithmetic::Degrees(op) => const_eval_float!(op.input; num::Float::to_degrees),
        Arithmetic::Radians(op) => const_eval_float!(op.input; num::Float::to_radians),
        Arithmetic::ArcTan2(op) => {
            use ConstantValue::*;
            let ty = op.lhs.ty;
            if let (Some(lhs), Some(rhs)) = (op.lhs.as_const(), op.rhs.as_const()) {
                let rhs = rhs.cast_to(ty);
                Some(match (lhs, rhs) {
                    (Float(lhs), Float(rhs)) => ConstantValue::Float(lhs.atan2(rhs)),
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Arithmetic::Sqrt(op) => const_eval_float!(op.input; num::Float::sqrt),
        Arithmetic::InverseSqrt(op) => {
            let sqrt = const_eval_float!(op.input; num::Float::sqrt)?;
            let ConstantValue::Float(val) = sqrt else {
                unreachable!()
            };
            Some(ConstantValue::Float(1.0 / val))
        }
        // `Round` ties to even on every backend (PTX `rni`, C++ `rint`, WGSL `round`,
        // MLIR `roundeven`), so constant folding must do the same.
        Arithmetic::Round(op) => const_eval_float!(op.input; f64::round_ties_even),
        Arithmetic::Floor(op) => const_eval_float!(op.input; num::Float::floor),
        Arithmetic::Ceil(op) => const_eval_float!(op.input; num::Float::ceil),
        Arithmetic::Trunc(op) => const_eval_float!(op.input; num::Float::trunc),
        Arithmetic::Recip(op) => const_eval_float!(op.input; num::Float::recip),
        Arithmetic::Neg(op) => {
            use ConstantValue::*;
            op.input.as_const().map(|input| match input {
                Int(input) => ConstantValue::Int(-input),
                Float(input) => ConstantValue::Float(-input),
                _ => unreachable!(),
            })
        }

        Arithmetic::Fma(op) => {
            use ConstantValue::*;
            let ty = op.a.ty;
            let a = op.a.as_const();
            let b = op.b.as_const();
            let c = op.c.as_const();

            a.zip(b).zip(c).map(|((a, b), c)| {
                let b = b.cast_to(ty);
                let c = c.cast_to(ty);
                match (a, b, c) {
                    (Float(a), Float(b), Float(c)) => ConstantValue::Float(a * b + c),
                    _ => unreachable!(),
                }
            })
        }
        Arithmetic::Clamp(op) => {
            use ConstantValue::*;
            let ty = op.input.ty;
            let a = op.input.as_const();
            let b = op.min_value.as_const();
            let c = op.max_value.as_const();

            a.zip(b).zip(c).map(|((a, b), c)| {
                let b = b.cast_to(ty);
                let c = c.cast_to(ty);
                match (a, b, c) {
                    (Int(a), Int(b), Int(c)) => ConstantValue::Int(a.clamp(b, c)),
                    (Float(a), Float(b), Float(c)) => ConstantValue::Float(a.clamp(b, c)),
                    (UInt(a), UInt(b), UInt(c)) => ConstantValue::UInt(a.clamp(b, c)),
                    _ => unreachable!(),
                }
            })
        }
        Arithmetic::Erf(_)
        | Arithmetic::Hypot(_)
        | Arithmetic::Rhypot(_)
        | Arithmetic::Magnitude(_)
        | Arithmetic::Normalize(_)
        | Arithmetic::VectorSum(_) => None,
    }
}

fn try_const_eval_cmp(op: &mut Comparison) -> Option<ConstantValue> {
    match op {
        Comparison::Equal(op) => const_eval_cmp!(== op.lhs, op.rhs),
        Comparison::NotEqual(op) => const_eval_cmp!(!= op.lhs, op.rhs),
        Comparison::Lower(op) => const_eval_cmp!(< op.lhs, op.rhs),
        Comparison::Greater(op) => const_eval_cmp!(> op.lhs, op.rhs),
        Comparison::LowerEqual(op) => const_eval_cmp!(<= op.lhs, op.rhs),
        Comparison::GreaterEqual(op) => const_eval_cmp!(>= op.lhs, op.rhs),
        Comparison::IsNan(op) => {
            use ConstantValue::*;
            op.input.as_const().map(|input| match input {
                Float(val) => Bool(val.is_nan()),
                // Integers, bools, uints can't be NaN, so always false
                Int(_) | UInt(_) | Bool(_) => Bool(false),
            })
        }
        Comparison::IsInf(op) => {
            use ConstantValue::*;
            op.input.as_const().map(|input| match input {
                Float(val) => Bool(val.is_infinite()),
                // Integers, bools, uints can't be infinite, so always false
                Int(_) | UInt(_) | Bool(_) => Bool(false),
            })
        }
    }
}

fn try_const_eval_bitwise(op: &mut Bitwise) -> Option<ConstantValue> {
    match op {
        Bitwise::BitwiseAnd(op) => const_eval_int!(&op.lhs, op.rhs),
        Bitwise::BitwiseOr(op) => const_eval_int!(| op.lhs, op.rhs),
        Bitwise::BitwiseXor(op) => const_eval_int!(^ op.lhs, op.rhs),
        Bitwise::ShiftLeft(op) => const_eval_int!(<< op.lhs, op.rhs),
        Bitwise::ShiftRight(op) => const_eval_int!(>> op.lhs, op.rhs),
        Bitwise::BitwiseNot(op) => {
            use ConstantValue::*;
            op.input.as_const().map(|input| match input {
                Int(input) => ConstantValue::Int(!input),
                UInt(input) => ConstantValue::UInt(!input),
                _ => unreachable!(),
            })
        }
        Bitwise::CountOnes(op) => {
            use ConstantValue::*;
            op.input.as_const().map(|input| match input {
                Int(input) => ConstantValue::Int(input.count_ones() as i64),
                UInt(input) => ConstantValue::UInt(input.count_ones() as u64),
                _ => unreachable!(),
            })
        }
        Bitwise::ReverseBits(op) => {
            use ConstantValue::*;
            op.input.as_const().map(|input| match input {
                Int(input) => ConstantValue::Int(input.reverse_bits()),
                UInt(input) => ConstantValue::UInt(input.reverse_bits()),
                _ => unreachable!(),
            })
        }
        Bitwise::LeadingZeros(_) | Bitwise::TrailingZeros(_) | Bitwise::FindFirstSet(_) => {
            // Depends too much on type width and Rust semantics, leave this one out of const eval
            None
        }
    }
}

fn try_const_eval_operator(op: &mut Operator, out_ty: Option<Type>) -> Option<ConstantValue> {
    match op {
        Operator::And(op) => const_eval_bool!(&&op.lhs, op.rhs),
        Operator::Or(op) => const_eval_bool!(|| op.lhs, op.rhs),
        Operator::Not(op) => {
            use ConstantValue::*;
            op.input.as_const().map(|input| match input {
                Bool(input) => ConstantValue::Bool(!input),
                _ => unreachable!(),
            })
        }
        Operator::Cast(op) => op.input.as_const().map(|val| val.cast_to(out_ty.unwrap())),
        Operator::Index(_)
        | Operator::NativeAddress(_)
        | Operator::NativeLoad(_)
        | Operator::NativeStore(_)
        | Operator::CopyMemory(_)
        | Operator::CopyMemoryBulk(_)
        | Operator::UncheckedIndex(_)
        | Operator::IndexAssign(_)
        | Operator::InitVector(_)
        | Operator::UncheckedIndexAssign(_)
        | Operator::Reinterpret(_)
        | Operator::Select(_) => None,
    }
}

#[cfg(test)]
mod modulo_tests {
    use super::*;
    use crate::optimizer::BasicBlock;
    use ruda_core::ir::{BinaryOperator, ElemType, FloatKind, IntKind, UIntKind};

    fn optimize_modulo(lhs: Variable, rhs: Variable) -> Operation {
        let mut opt = Optimizer::default();
        let block = opt.program.add_node(BasicBlock::default());
        let out = Variable::new(VariableKind::LocalConst { id: 0 }, lhs.ty);
        opt.program[block].ops.borrow_mut().push(Instruction::new(
            Arithmetic::Modulo(BinaryOperator { lhs, rhs }), out,
        ));
        ConstOperandSimplify.apply_post_ssa(&mut opt, AtomicCounter::default());
        ConstEval.apply_post_ssa(&mut opt, AtomicCounter::default());
        let operation = opt.program[block].ops.borrow()[0].operation.clone();
        operation
    }

    #[test]
    fn float_modulo_keeps_fractional_and_special_values() {
        for kind in [FloatKind::F32, FloatKind::F64] {
            let ty = Type::scalar(ElemType::Float(kind));
            for (lhs, rhs) in [(1.5, 1.0), (-1.5, 1.0), (-0.0, 1.0),
                (0.0, 0.0), (0.0, f64::NAN), (f64::INFINITY, 1.0)] {
                let Operation::Copy(value) = optimize_modulo(
                    Variable::constant(ConstantValue::Float(lhs), ty),
                    Variable::constant(ConstantValue::Float(rhs), ty),
                ) else { panic!("constant modulo was not evaluated") };
                let Some(ConstantValue::Float(actual)) = value.as_const() else {
                    panic!("float modulo produced a non-float constant")
                };
                let expected = lhs % rhs;
                assert!((expected.is_nan() && actual.is_nan())
                    || actual.to_bits() == expected.to_bits());
            }
        }
    }

    #[test]
    fn dynamic_float_modulo_by_one_is_not_zero() {
        let ty = Type::scalar(ElemType::Float(FloatKind::F32));
        assert!(matches!(optimize_modulo(
            Variable::new(VariableKind::LocalConst { id: 1 }, ty),
            Variable::constant(ConstantValue::Float(1.0), ty),
        ), Operation::Arithmetic(Arithmetic::Modulo(_))));
    }

    #[test]
    fn integer_modulo_by_one_still_simplifies() {
        for ty in [Type::scalar(ElemType::Int(IntKind::I32)),
            Type::scalar(ElemType::UInt(UIntKind::U32))] {
            let Operation::Copy(value) = optimize_modulo(
                Variable::new(VariableKind::LocalConst { id: 1 }, ty),
                ty.constant(ConstantValue::Int(1)),
            ) else { panic!("integer modulo was not simplified") };
            assert!(value.is_constant(0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruda_core::ir::{BinaryOperator, ElemType, FloatKind, IntKind, UnaryOperator};

    const SPECIAL: [f64; 8] = [1.5, -1.5, 0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 2.5];

    fn f32_ty() -> Type {
        Type::scalar(ElemType::Float(FloatKind::F32))
    }

    fn i32_ty() -> Type {
        Type::scalar(ElemType::Int(IntKind::I32))
    }

    fn local(ty: Type) -> Variable {
        Variable::new(VariableKind::LocalConst { id: 0 }, ty)
    }

    fn binary(op: fn(BinaryOperator) -> Arithmetic, lhs: Variable, rhs: Variable) -> Instruction {
        let out = Variable::new(VariableKind::LocalConst { id: 1 }, lhs.ty);
        Instruction::new(Operation::Arithmetic(op(BinaryOperator { lhs, rhs })), out)
    }

    fn eval_float(op: &Arithmetic, x: f64) -> f64 {
        let value = |var: &Variable| match var.kind {
            VariableKind::Constant(ConstantValue::Float(val)) => val,
            _ => x,
        };
        match op {
            Arithmetic::Add(op) => value(&op.lhs) + value(&op.rhs),
            Arithmetic::Sub(op) => value(&op.lhs) - value(&op.rhs),
            Arithmetic::Mul(op) => value(&op.lhs) * value(&op.rhs),
            Arithmetic::Div(op) => value(&op.lhs) / value(&op.rhs),
            Arithmetic::Modulo(op) => value(&op.lhs) % value(&op.rhs),
            _ => unreachable!(),
        }
    }

    fn same_float(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
    }

    /// Every float rewrite must produce the same value as the original expression for all
    /// special inputs; rules that cannot guarantee this must leave the instruction alone.
    #[test]
    fn float_operand_simplification_preserves_results() {
        let ops: [fn(BinaryOperator) -> Arithmetic; 5] =
            [Arithmetic::Add, Arithmetic::Sub, Arithmetic::Mul, Arithmetic::Div, Arithmetic::Modulo];
        let constants = [0.0, -0.0, 1.0];
        let x = local(f32_ty());
        for op in ops {
            for constant in constants {
                let c = Variable::constant(ConstantValue::Float(constant), f32_ty());
                for (lhs, rhs) in [(c, x), (x, c)] {
                    let mut inst = binary(op, lhs, rhs);
                    let Operation::Arithmetic(original) = inst.operation.clone() else {
                        unreachable!()
                    };
                    simplify_const_operand(&mut inst, &AtomicCounter::new(0));
                    let Operation::Copy(result) = inst.operation else {
                        continue;
                    };
                    for input in SPECIAL {
                        let expected = eval_float(&original, input);
                        let actual = match result.kind {
                            VariableKind::Constant(ConstantValue::Float(val)) => val,
                            VariableKind::Constant(ConstantValue::Int(val)) => val as f64,
                            _ => input,
                        };
                        assert!(
                            same_float(expected, actual),
                            "{original} rewritten to {result} changes x={input}: {expected} vs {actual}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn float_modulo_by_one_is_not_folded_to_zero() {
        let one = Variable::constant(ConstantValue::Float(1.0), f32_ty());
        let mut inst = binary(Arithmetic::Modulo, local(f32_ty()), one);
        simplify_const_operand(&mut inst, &AtomicCounter::new(0));
        assert!(matches!(inst.operation, Operation::Arithmetic(Arithmetic::Modulo(_))));
    }

    #[test]
    fn integer_absorbing_rules_still_apply() {
        let zero = Variable::constant(ConstantValue::Int(0), i32_ty());
        let one = Variable::constant(ConstantValue::Int(1), i32_ty());
        let x = local(i32_ty());
        for mut inst in [
            binary(Arithmetic::Mul, zero, x),
            binary(Arithmetic::Div, zero, x),
            binary(Arithmetic::Modulo, x, one),
            binary(Arithmetic::Add, x, zero),
        ] {
            let changes = AtomicCounter::new(0);
            simplify_const_operand(&mut inst, &changes);
            assert_eq!(changes.get(), 1, "{inst}");
        }
    }

    #[test]
    fn round_constant_matches_runtime_ties_to_even() {
        for (input, expected) in [(0.5, 0.0), (1.5, 2.0), (2.5, 2.0), (-0.5, -0.0), (-1.5, -2.0), (-2.5, -2.0)] {
            let input = Variable::constant(ConstantValue::Float(input), f32_ty());
            let mut op = Arithmetic::Round(UnaryOperator { input });
            let Some(ConstantValue::Float(actual)) = try_const_eval_arithmetic(&mut op) else {
                panic!("round was not folded");
            };
            assert_eq!(actual.to_bits(), f64::to_bits(expected), "round({input})");
        }
    }

    #[test]
    fn float_remainder_constant_takes_divisor_sign() {
        for (lhs, rhs, expected) in [(-1.0, 3.0, 2.0), (5.5, -2.0, -0.5), (7.0, 2.5, 2.0), (-6.0, 3.0, -0.0)] {
            let lhs = Variable::constant(ConstantValue::Float(lhs), f32_ty());
            let rhs = Variable::constant(ConstantValue::Float(rhs), f32_ty());
            let mut op = Arithmetic::Remainder(BinaryOperator { lhs, rhs });
            let Some(ConstantValue::Float(actual)) = try_const_eval_arithmetic(&mut op) else {
                panic!("remainder was not folded");
            };
            assert_eq!(actual.to_bits(), f64::to_bits(expected), "{lhs} rem {rhs}");
        }
    }

    #[test]
    fn integer_division_by_constant_zero_is_not_folded() {
        let lhs = Variable::constant(ConstantValue::Int(7), i32_ty());
        let rhs = Variable::constant(ConstantValue::Int(0), i32_ty());
        for op in [Arithmetic::Div, Arithmetic::Modulo, Arithmetic::Remainder] {
            assert!(try_const_eval_arithmetic(&mut op(BinaryOperator { lhs, rhs })).is_none());
        }
    }
}
