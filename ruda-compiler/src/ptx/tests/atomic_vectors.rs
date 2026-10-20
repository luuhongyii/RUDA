use super::*;

fn atomic_kernel(kind: FloatKind, width: usize, arithmetic: bool) -> KernelDefinition {
    let mut k = kernel(&format!("atomic_complete_{kind:?}_{width}_{arithmetic}"));
    let ty = Type::from(kind).with_vector_size(width);
    let atomic_ty = Type::new(StorageType::Atomic(ElemType::Float(kind))).with_vector_size(width);
    let array = buffer(&mut k, atomic_ty, Visibility::ReadWrite);
    let output = buffer(&mut k, ty, Visibility::ReadWrite);
    let pointer = local(0, atomic_ty);
    load(&mut k.body, array, 0u32.into(), pointer);
    let value = Variable::constant(ConstantValue::Float(2.0), ty);
    let binary = BinaryOperator {
        lhs: pointer,
        rhs: value,
    };
    let mut operations = vec![
        AtomicOp::Load(UnaryOperator { input: pointer }),
        AtomicOp::Swap(binary.clone()),
        AtomicOp::CompareAndSwap(CompareAndSwapOperator {
            input: pointer,
            cmp: value,
            val: value,
        }),
    ];
    if arithmetic {
        operations.extend([AtomicOp::Add(binary.clone()), AtomicOp::Sub(binary.clone())]);
        if matches!(kind, FloatKind::F16 | FloatKind::BF16) && width > 1 {
            operations.extend([AtomicOp::Min(binary.clone()), AtomicOp::Max(binary)]);
        }
    }
    for (index, operation) in operations.into_iter().enumerate() {
        let result = local(index as u32 + 1, ty);
        k.body.register(Instruction::new(operation, result));
        store(&mut k.body, output, (index as u32).into(), result);
    }
    k.body.register(Instruction::new(
        AtomicOp::Store(UnaryOperator { input: value }),
        pointer,
    ));
    k
}

#[test]
fn half_vectors_use_native_reductions_and_whole_value_exchange() {
    let options = PtxCompilationOptions {
        target: Some(PtxTarget {
            sm: 90,
            version: (8, 8),
        }),
    };
    for kind in [FloatKind::F16, FloatKind::BF16] {
        for width in [2, 4, 8] {
            let compiled = PtxCompiler
                .compile(
                    atomic_kernel(kind, width, true),
                    &options,
                    ExecutionMode::Checked,
                    UIntKind::U32.into(),
                )
                .unwrap();
            let source = &compiled.source;
            for op in ["add", "min", "max"] {
                assert!(source.contains(&format!("atom.{op}.noftz.v{width}.")));
            }
            assert_eq!(
                source
                    .matches(&format!("atom.cas.b{} ", width * 16))
                    .count(),
                2
            );
            assert_eq!(
                source
                    .matches(&format!("atom.exch.b{} ", width * 16))
                    .count(),
                2
            );
            export(&compiled);
        }
    }
}

#[test]
fn float_atomic_loads_are_bit_preserving() {
    let options = PtxCompilationOptions {
        target: Some(PtxTarget {
            sm: 90,
            version: (8, 8),
        }),
    };
    for (kind, widths) in [
        (FloatKind::F16, vec![1]),
        (FloatKind::BF16, vec![1]),
        (FloatKind::F32, vec![1, 2, 4]),
        (FloatKind::F64, vec![1, 2]),
    ] {
        for width in widths {
            let compiled = PtxCompiler
                .compile(
                    atomic_kernel(kind, width, false),
                    &options,
                    ExecutionMode::Checked,
                    UIntKind::U32.into(),
                )
                .unwrap();
            assert!(!compiled.source.contains("atom.add"));
            assert!(compiled.source.contains("atom.cas"));
            export(&compiled);
        }
    }
}

#[test]
fn whole_128_bit_exchange_requires_matching_target() {
    for target in [
        PtxTarget {
            sm: 89,
            version: (8, 8),
        },
        PtxTarget {
            sm: 90,
            version: (8, 1),
        },
    ] {
        let result = PtxCompiler.compile(
            atomic_kernel(FloatKind::F32, 4, false),
            &PtxCompilationOptions {
                target: Some(target),
            },
            ExecutionMode::Unchecked,
            UIntKind::U32.into(),
        );
        assert!(format!("{:?}", result.unwrap_err()).contains("128-bit atomics require"));
    }
}

#[test]
fn vector_add_dispatches_actual_pointer_space_after_reassignment() {
    for (kind, width) in [
        (FloatKind::F16, 8),
        (FloatKind::BF16, 8),
        (FloatKind::F32, 4),
    ] {
        for mode in 0..4 {
            let mut k = kernel(&format!("atomic_shared_{kind:?}_{mode}"));
            let ty = Type::from(kind).with_vector_size(width);
            let atomic_ty =
                Type::new(StorageType::Atomic(ElemType::Float(kind))).with_vector_size(width);
            let global = buffer(&mut k, atomic_ty, Visibility::ReadWrite);
            let output = buffer(&mut k, ty, Visibility::ReadWrite);
            let shared = *k.body.create_shared_array(atomic_ty, 1, Some(16));
            let pointer = local(100, atomic_ty);
            let other = local(101, atomic_ty);
            let address = local(102, UIntKind::U64.into());
            match mode {
                0 => load(&mut k.body, shared, 0u32.into(), pointer),
                1 => {
                    k.body.register(Instruction::new(
                        Operator::NativeAddress(BinaryOperator {
                            lhs: shared,
                            rhs: 0u32.into(),
                        }),
                        address,
                    ));
                    k.body.register(Instruction::new(
                        Operator::Reinterpret(UnaryOperator { input: address }),
                        pointer,
                    ));
                }
                _ => {
                    load(&mut k.body, global, 0u32.into(), pointer);
                    load(&mut k.body, shared, 0u32.into(), other);
                    let operation: Operation = if mode == 2 {
                        Operation::Copy(other)
                    } else {
                        Operator::Select(Select {
                            cond: false.into(),
                            then: pointer,
                            or_else: other,
                        })
                        .into()
                    };
                    k.body.register(Instruction::new(operation, pointer));
                }
            }
            let value = Variable::constant(ConstantValue::Float(1.0), ty);
            for (index, operation) in [
                AtomicOp::Add(BinaryOperator {
                    lhs: pointer,
                    rhs: value,
                }),
                AtomicOp::Sub(BinaryOperator {
                    lhs: pointer,
                    rhs: value,
                }),
            ]
            .into_iter()
            .enumerate()
            {
                let result = local(103 + index as u32, ty);
                k.body.register(Instruction::new(operation, result));
                store(&mut k.body, output, (index as u32).into(), result);
            }
            let compiled = PtxCompiler
                .compile(
                    k,
                    &PtxCompilationOptions {
                        target: Some(PtxTarget {
                            sm: 90,
                            version: (8, 8),
                        }),
                    },
                    ExecutionMode::Unchecked,
                    UIntKind::U32.into(),
                )
                .unwrap();
            assert_eq!(compiled.source.matches("isspacep.global").count(), 2);
            assert!(compiled.source.contains("cvta.shared.u64"));
            let suffix = match kind {
                FloatKind::F16 => ".noftz.f16",
                FloatKind::BF16 => ".noftz.bf16",
                _ => ".f32",
            };
            assert_eq!(
                compiled.source.matches(&format!(".add{suffix} ")).count(),
                2 * width
            );
            export(&compiled);
        }
    }
}
