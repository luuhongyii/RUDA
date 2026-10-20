use melior::{
    dialect::{arith, llvm},
    ir::{
        Identifier,
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
    },
};
use ruda_core::ir::AtomicOp;

use crate::mlir::visitor::prelude::*;

impl<'a> Visitor<'a> {
    pub(crate) fn visit_atomic(&mut self, atomic: &AtomicOp, out: Variable) {
        let address = match atomic {
            AtomicOp::Load(op) => op.input,
            AtomicOp::Store(_) => out,
            AtomicOp::Swap(op) | AtomicOp::Add(op) | AtomicOp::Sub(op)
            | AtomicOp::Min(op) | AtomicOp::Max(op) | AtomicOp::And(op)
            | AtomicOp::Or(op) | AtomicOp::Xor(op) => op.lhs,
            AtomicOp::CompareAndSwap(op) => op.input,
        };
        assert!(address.storage_type().is_atomic(), "atomic access requires an atomic reference");
        assert_eq!(address.vector_size(), 1, "CPU atomics require a scalar pointee");
        let pointer = self.get_variable(address);
        let element = address.elem_type();
        let ty = element.to_type(self.context);
        let integer_attr = |value| {
            IntegerAttribute::new(IntegerType::new(self.context, 64).into(), value).into()
        };
        let monotonic = integer_attr(2);
        let alignment = integer_attr(element.size() as i64);
        let attributes = [
            (Identifier::new(self.context, "ordering"), monotonic),
            (Identifier::new(self.context, "alignment"), alignment),
        ];
        let previous = match atomic {
            AtomicOp::Load(_) => self.append_operation_with_result(
                OperationBuilder::new("llvm.load", self.location)
                    .add_operands(&[pointer])
                    .add_results(&[ty])
                    .add_attributes(&attributes)
                    .build().expect("valid atomic load"),
            ),
            AtomicOp::Store(op) => {
                let value = self.get_variable(op.input);
                self.block.append_operation(
                    OperationBuilder::new("llvm.store", self.location)
                        .add_operands(&[value, pointer])
                        .add_attributes(&attributes)
                        .build().expect("valid atomic store"),
                );
                return;
            }
            AtomicOp::CompareAndSwap(op) => {
                let mut cmp = self.get_variable(op.cmp);
                let mut value = self.get_variable(op.val);
                let compare_type = if element.is_float() {
                    let bits = IntegerType::new(self.context, element.size_bits() as u32).into();
                    cmp = self.append_operation_with_result(arith::bitcast(cmp, bits, self.location));
                    value = self.append_operation_with_result(arith::bitcast(value, bits, self.location));
                    bits
                } else {
                    ty
                };
                let result_type = llvm::r#type::r#struct(
                    self.context, &[compare_type, IntegerType::new(self.context, 1).into()], false,
                );
                let pair = self.append_operation_with_result(
                    OperationBuilder::new("llvm.cmpxchg", self.location)
                        .add_operands(&[pointer, cmp, value])
                        .add_results(&[result_type])
                        .add_attributes(&[
                            (Identifier::new(self.context, "success_ordering"), monotonic),
                            (Identifier::new(self.context, "failure_ordering"), monotonic),
                            (Identifier::new(self.context, "alignment"), alignment),
                        ])
                        .build().expect("valid atomic compare-and-swap"),
                );
                let previous = self.append_operation_with_result(llvm::extract_value(
                    self.context, pair, DenseI64ArrayAttribute::new(self.context, &[0]),
                    compare_type, self.location,
                ));
                if element.is_float() {
                    self.append_operation_with_result(arith::bitcast(previous, ty, self.location))
                } else {
                    previous
                }
            }
            AtomicOp::Swap(op) | AtomicOp::Add(op) | AtomicOp::Sub(op)
            | AtomicOp::Min(op) | AtomicOp::Max(op) | AtomicOp::And(op)
            | AtomicOp::Or(op) | AtomicOp::Xor(op) => {
                let kind = match atomic {
                    AtomicOp::Swap(_) => 0,
                    AtomicOp::Add(_) if element.is_float() => 11,
                    AtomicOp::Sub(_) if element.is_float() => 12,
                    AtomicOp::Max(_) if element.is_float() => 13,
                    AtomicOp::Min(_) if element.is_float() => 14,
                    AtomicOp::Add(_) => 1,
                    AtomicOp::Sub(_) => 2,
                    AtomicOp::And(_) => 3,
                    AtomicOp::Or(_) => 5,
                    AtomicOp::Xor(_) => 6,
                    AtomicOp::Max(_) if element.is_signed_int() => 7,
                    AtomicOp::Min(_) if element.is_signed_int() => 8,
                    AtomicOp::Max(_) => 9,
                    AtomicOp::Min(_) => 10,
                    _ => unreachable!(),
                };
                let value = self.get_variable(op.rhs);
                self.append_operation_with_result(
                    OperationBuilder::new("llvm.atomicrmw", self.location)
                        .add_operands(&[pointer, value])
                        .add_results(&[ty])
                        .add_attributes(&attributes)
                        .add_attributes(&[(Identifier::new(self.context, "bin_op"), integer_attr(kind))])
                        .build().expect("valid atomic read-modify-write"),
                )
            }
        };
        self.insert_variable(out, previous);
    }
}
