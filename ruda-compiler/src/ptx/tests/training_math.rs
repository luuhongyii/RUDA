use super::*;

#[test]
fn inverse_sqrt_reduced_precision_is_explicit_and_f32_only() {
    for kind in [FloatKind::F32, FloatKind::F64] {
        for reduced in [false, true] {
            let ty=Type::scalar(ElemType::Float(kind));let mut k=kernel("normalization_rsqrt");
            let input=buffer(&mut k,ty,Visibility::Read);let output=buffer(&mut k,ty,Visibility::ReadWrite);
            let index=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U32.into());let x=local(0,ty);let y=local(1,ty);
            load(&mut k.body,input,index,x);
            let mut op=Instruction::new(Arithmetic::InverseSqrt(UnaryOperator{input:x}),y);
            if reduced {op.modes.fp_math_mode=FastMath::ReducedPrecision.into();}
            k.body.instructions.push(op);store(&mut k.body,output,index,y);
            let ptx=compile(k,ExecutionMode::Checked,UIntKind::U32).unwrap();
            assert_eq!(ptx.source.contains("rsqrt.approx.f32"),reduced && kind==FloatKind::F32);
            if !reduced || kind==FloatKind::F64 {assert!(ptx.source.contains("sqrt.rn.f64"));}
        }
    }
}

#[test]
fn double_error_function_compiles() {
    let ty=Type::scalar(ElemType::Float(FloatKind::F64));
    let mut k=kernel("error_erf_f64");
    let input=buffer(&mut k,ty,Visibility::Read);
    let output=buffer(&mut k,ty,Visibility::ReadWrite);
    let index=Variable::builtin(Builtin::AbsolutePosX,UIntKind::U32.into());
    let x=local(0,ty);let result=local(1,ty);
    load(&mut k.body,input,index,x);
    k.body.instructions.push(Instruction::new(Arithmetic::Erf(UnaryOperator{input:x}),result));
    store(&mut k.body,output,index,result);
    let ptx=compile(k,ExecutionMode::Checked,UIntKind::U32).unwrap();
    assert!(!ptx.source.contains("call.uni"));
    assert!(!ptx.source.contains("cvt.rn.f32.f64"));
    assert!(ptx.source.contains("fma.rn.f64"));
    export(&ptx);
}

#[test]
fn hyperbolic_and_inverse_trig_compile() {
    for kind in [FloatKind::F16, FloatKind::BF16, FloatKind::F32, FloatKind::F64] {
        let ty = Type::scalar(ElemType::Float(kind));
        for operation in ["sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "asin", "acos", "atan", "tan"] {
            if kind != FloatKind::F64 && !matches!(operation, "asin" | "acos" | "atan" | "tan") { continue; }
            let suffix = match kind { FloatKind::F16 => "f16", FloatKind::BF16 => "bf16", FloatKind::F32 => "f32", _ => "f64" };
            let mut k = kernel(&format!("hyperbolic_{operation}_{suffix}"));
            let input = buffer(&mut k, ty, Visibility::Read);
            let output = buffer(&mut k, ty, Visibility::ReadWrite);
            let index = Variable::builtin(Builtin::AbsolutePosX, UIntKind::U32.into());
            let x = local(0, ty);
            let result = local(1, ty);
            load(&mut k.body, input, index, x);
            let arg = UnaryOperator { input: x };
            let operation = match operation {
                "sinh" => Arithmetic::Sinh(arg), "cosh" => Arithmetic::Cosh(arg),
                "tanh" => Arithmetic::Tanh(arg), "asinh" => Arithmetic::ArcSinh(arg),
                "acosh" => Arithmetic::ArcCosh(arg), "atanh" => Arithmetic::ArcTanh(arg),
                "asin" => Arithmetic::ArcSin(arg), "acos" => Arithmetic::ArcCos(arg),
                "atan" => Arithmetic::ArcTan(arg), _ => Arithmetic::Tan(arg),
            };
            k.body.instructions.push(Instruction::new(operation, result));
            store(&mut k.body, output, index, result);
            let ptx = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
            assert!(!ptx.source.contains("call.uni"));
            if kind == FloatKind::F64 { assert!(!ptx.source.contains("cvt.rn.f32.f64")); }
            export(&ptx);
        }
    }
}

#[test]
fn double_transcendentals_compile() {
    let ty = Type::scalar(ElemType::Float(FloatKind::F64));
    for operation in ["exp", "log", "log1p", "pow", "sin", "cos"] {
        let mut k = kernel(&format!("double_{operation}"));
        let a = buffer(&mut k, ty, Visibility::Read);
        let b = buffer(&mut k, ty, Visibility::Read);
        let output = buffer(&mut k, ty, Visibility::ReadWrite);
        let idx = Variable::builtin(Builtin::AbsolutePosX, UIntKind::U32.into());
        let x = local(0, ty);
        let y = local(1, ty);
        let z = local(2, ty);
        load(&mut k.body, a, idx, x);
        load(&mut k.body, b, idx, y);
        let unary = UnaryOperator { input: x };
        let op = match operation {
            "exp" => Arithmetic::Exp(unary), "log" => Arithmetic::Log(unary),
            "log1p" => Arithmetic::Log1p(unary), "sin" => Arithmetic::Sin(unary),
            "cos" => Arithmetic::Cos(unary), _ => Arithmetic::Powf(BinaryOperator { lhs: x, rhs: y }),
        };
        k.body.instructions.push(Instruction::new(op, z));
        store(&mut k.body, output, idx, z);
        let result = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
        assert!(!result.source.contains("call.uni"));
        assert!(!result.source.contains("cvt.rn.f32.f64"));
        assert!(result.source.contains("fma.rn.f64"));
        export(&result);
    }
}

#[test]
fn statistics_binary_math_compiles() {
    for kind in [FloatKind::F16, FloatKind::BF16, FloatKind::F32, FloatKind::F64] {
        let ty = Type::scalar(ElemType::Float(kind));
        for angle in [false, true] {
            let mut k = kernel(&format!("statistics_{}_{}", if angle { "atan2" } else { "hypot" },
                match kind { FloatKind::F16 => "f16", FloatKind::BF16 => "bf16", FloatKind::F32 => "f32", _ => "f64" }));
            let a = buffer(&mut k, ty, Visibility::Read);
            let b = buffer(&mut k, ty, Visibility::Read);
            let output = buffer(&mut k, ty, Visibility::ReadWrite);
            let idx = Variable::builtin(Builtin::AbsolutePosX, UIntKind::U32.into());
            let x = local(0, ty);
            let y = local(1, ty);
            let z = local(2, ty);
            load(&mut k.body, a, idx, x);
            load(&mut k.body, b, idx, y);
            let op = BinaryOperator { lhs: x, rhs: y };
            k.body.instructions.push(Instruction::new(
                if angle { Arithmetic::ArcTan2(op) } else { Arithmetic::Hypot(op) }, z));
            store(&mut k.body, output, idx, z);
            let result = compile(k, ExecutionMode::Checked, UIntKind::U32).unwrap();
            assert!(!result.source.contains("call.uni"));
            assert!(result.source.contains(if angle { "fma.rn.f64" } else { "sqrt.rn.f64" }));
            export(&result);
        }
    }
}

#[test]
fn erf_and_powf_compile_for_training_storage() {
    for kind in [FloatKind::F32, FloatKind::F16, FloatKind::BF16] {
        let ty = Type::scalar(ElemType::Float(kind));
        for power in [false, true] {
            let mut k = kernel(&format!("training_math_{kind:?}_{power}"));
            let input = buffer(&mut k, ty, Visibility::Read);
            let exponent = buffer(&mut k, ty, Visibility::Read);
            let output = buffer(&mut k, ty, Visibility::ReadWrite);
            let idx = Variable::builtin(Builtin::AbsolutePosX, UIntKind::U32.into());
            let x = local(0, ty);
            let y = local(1, ty);
            let z = local(2, ty);
            load(&mut k.body, input, idx, x);
            load(&mut k.body, exponent, idx, y);
            let op = if power { Arithmetic::Powf(BinaryOperator { lhs: x, rhs: y }) }
                else { Arithmetic::Erf(UnaryOperator { input: x }) };
            k.body.instructions.push(Instruction::new(op, z));
            store(&mut k.body, output, idx, z);
            let mut target = options();
            if kind != FloatKind::BF16 { target.target.as_mut().unwrap().sm = 75; }
            let result = PtxCompiler.compile(k, &target, ExecutionMode::Checked, UIntKind::U32.into()).unwrap();
            assert!(!result.source.contains("call.uni"));
            assert!(result.source.contains(if power { "mul.rn.f64" } else { "fma.rn.f32" }));
            export(&result);
        }
    }
}
