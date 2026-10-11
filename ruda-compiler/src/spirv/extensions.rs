use rspirv::spirv::Word;

use crate::spirv::SpirvCompiler;

use super::{GLCompute, SpirvTarget};

pub trait TargetExtensions<T: SpirvTarget> {
    fn round(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn f_abs(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn s_abs(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn floor(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn ceil(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn trunc(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn sin(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn cos(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn tan(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn tanh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn sinh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn cosh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn asin(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn acos(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn atan(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn asinh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn acosh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn atanh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn degrees(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn radians(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn atan2(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word);
    fn pow(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word);
    fn exp(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn log(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn sqrt(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn inverse_sqrt(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn f_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word);
    fn u_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word);
    fn s_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word);
    fn f_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word);
    fn u_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word);
    fn s_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word);
    fn f_clamp(b: &mut SpirvCompiler<T>, ty: Word, input: Word, min: Word, max: Word, out: Word);
    fn u_clamp(b: &mut SpirvCompiler<T>, ty: Word, input: Word, min: Word, max: Word, out: Word);
    fn s_clamp(b: &mut SpirvCompiler<T>, ty: Word, input: Word, min: Word, max: Word, out: Word);
    fn magnitude(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn normalize(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);

    fn find_msb(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
    fn find_lsb(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word);
}

pub mod glcompute {

    use super::*;
    use rspirv::{dr::Operand, spirv::GlslStd450Op};

    fn glsl<T: SpirvTarget>(b: &mut SpirvCompiler<T>, ty: Word, out: Word, opcode: GlslStd450Op, args: &[Word]) {
        let set = super::super::extended::import(&mut b.builder, "GLSL.std.450");
        b.ext_inst(ty, Some(out), set, opcode as u32, args.iter().copied().map(Operand::IdRef)).unwrap();
    }

    impl<T: SpirvTarget> TargetExtensions<T> for GLCompute {
        fn round(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::RoundEven, &[input]);
        }

        fn f_abs(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::FAbs, &[input]);
        }

        fn s_abs(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::SAbs, &[input]);
        }

        fn floor(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Floor, &[input]);
        }

        fn ceil(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Ceil, &[input]);
        }

        fn trunc(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Trunc, &[input]);
        }

        fn sin(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Sin, &[input]);
        }

        fn cos(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Cos, &[input]);
        }

        fn tan(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Tan, &[input]);
        }

        fn tanh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Tanh, &[input]);
        }

        fn sinh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Sinh, &[input]);
        }

        fn cosh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Cosh, &[input]);
        }

        fn asin(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Asin, &[input]);
        }

        fn acos(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Acos, &[input]);
        }

        fn atan(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Atan, &[input]);
        }

        fn asinh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Asinh, &[input]);
        }

        fn acosh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Acosh, &[input]);
        }

        fn atanh(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Atanh, &[input]);
        }

        fn degrees(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Degrees, &[input]);
        }

        fn radians(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Radians, &[input]);
        }

        fn atan2(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Atan2, &[lhs, rhs]);
        }

        fn pow(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Pow, &[lhs, rhs]);
        }
        fn exp(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Exp, &[input]);
        }

        fn log(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Log, &[input]);
        }

        fn sqrt(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Sqrt, &[input]);
        }

        fn inverse_sqrt(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::InverseSqrt, &[input]);
        }

        fn f_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::FMin, &[lhs, rhs]);
        }

        fn u_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::UMin, &[lhs, rhs]);
        }

        fn s_min(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::SMin, &[lhs, rhs]);
        }

        fn f_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::FMax, &[lhs, rhs]);
        }

        fn u_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::UMax, &[lhs, rhs]);
        }

        fn s_max(b: &mut SpirvCompiler<T>, ty: Word, lhs: Word, rhs: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::SMax, &[lhs, rhs]);
        }

        fn f_clamp(
            b: &mut SpirvCompiler<T>,
            ty: Word,
            input: Word,
            min: Word,
            max: Word,
            out: Word,
        ) {
            glsl(b, ty, out, GlslStd450Op::FClamp, &[input, min, max]);
        }

        fn u_clamp(
            b: &mut SpirvCompiler<T>,
            ty: Word,
            input: Word,
            min: Word,
            max: Word,
            out: Word,
        ) {
            glsl(b, ty, out, GlslStd450Op::UClamp, &[input, min, max]);
        }

        fn s_clamp(
            b: &mut SpirvCompiler<T>,
            ty: Word,
            input: Word,
            min: Word,
            max: Word,
            out: Word,
        ) {
            glsl(b, ty, out, GlslStd450Op::SClamp, &[input, min, max]);
        }

        fn magnitude(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Length, &[input]);
        }

        fn normalize(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::Normalize, &[input]);
        }

        fn find_msb(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::FindUMsb, &[input]);
        }

        fn find_lsb(b: &mut SpirvCompiler<T>, ty: Word, input: Word, out: Word) {
            glsl(b, ty, out, GlslStd450Op::FindILsb, &[input]);
        }
    }
}
