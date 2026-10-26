use crate::dsl::prelude::*;
use alloc::vec;
use ruda_core::ir::{CoopMma, FloatKind, Operation, PointerOp, Type, UIntKind};

#[ruda]
fn vectors(a: &Vector<f16, Const<8>>, b: &Vector<f16, Const<4>>, c: &mut Vector<f32, Const<4>>) {
    let definition = super::MmaDefinition::<f16, f16, f32>::new(16usize, 8usize, 16usize);
    definition.execute_vectors_inplace(a, b, c);
}

#[ruda]
fn arrays(a: &Array<Vector<f16, Const<2>>>, b: &Array<Vector<f16, Const<2>>>, c: &mut Vector<f32, Const<4>>) {
    let definition = super::MmaDefinition::<f16, f16, f32>::new(16usize, 8usize, 16usize);
    definition.execute_vector_inplace(a, b, c);
}

#[test]
fn manual_mma_vector_intrinsics_read_values_and_write_through_accumulator_reference() {
    for vector_operands in [false, true] {
        let mut scope = Scope::root(false);
        scope.register_type::<usize>(UIntKind::U32.into());
        let half = Type::from(FloatKind::F16);
        let c = scope.create_local(Type::from(FloatKind::F32).with_vector_size(4));
        let c_reference = scope.reference_borrow(c, true).unwrap();
        let c_pointer = scope.reference_pointer(*c_reference).unwrap();
        if vector_operands {
            let a = scope.create_local(half.with_vector_size(8));
            let b = scope.create_local(half.with_vector_size(4));
            let a = scope.reference_borrow(a, false).unwrap();
            let b = scope.reference_borrow(b, false).unwrap();
            vectors::expand(&mut scope, a.into(), b.into(), c_reference.into());
        } else {
            let a = scope.create_local_array(half.with_vector_size(2), 4);
            let b = scope.create_local_array(half.with_vector_size(2), 2);
            arrays::expand(&mut scope, a.into(), b.into(), c_reference.into());
        }
        let (position, result) = scope.instructions.iter().enumerate().find_map(|(i, instruction)| {
            if let Operation::CoopMma(CoopMma::ExecuteManual { registers_a, registers_b, registers_c, .. }) = instruction.operation {
                for value in [registers_a, registers_b, registers_c] {
                    assert!(scope.reference_target(value).is_none(), "MMA must read a value, not a reference proxy");
                }
                Some((i, instruction.out.unwrap()))
            } else { None }
        }).expect("manual MMA instruction");
        let mut values = vec![result];
        let mut writeback = false;
        for instruction in &scope.instructions[position + 1..] {
            match instruction.operation {
                Operation::Copy(input) if values.contains(&input) => values.push(instruction.out.unwrap()),
                Operation::Pointer(PointerOp::Store { pointer, value, .. })
                    if pointer == c_pointer && values.contains(&value) => writeback = true,
                _ => {}
            }
        }
        assert!(writeback, "MMA result must update the caller's accumulator, not a discarded proxy");
    }
}
