use melior::{
    dialect::{arith, index, llvm, memref, ods::vector, scf},
    ir::{Block, Identifier, Region, attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType},
};
use ruda_core::ir::{BinaryOperator, CopyMemoryBulkOperator, IndexOperator, Synchronization, UnaryOperator, VariableKind};

use crate::mlir::visitor::prelude::*;

impl<'a> Visitor<'a> {
    pub(crate) fn visit_workgroup_uniform_load(&mut self, op: &IndexOperator, out: Variable) {
        assert!(matches!(op.list.kind, VariableKind::SharedArray { .. } | VariableKind::Shared { .. }));
        let width = if op.vector_size == 0 { op.list.vector_size() } else { op.vector_size };
        assert_eq!(out.vector_size(), width);
        assert_eq!(out.elem_type(), op.list.elem_type());
        self.visit_synchronization(&Synchronization::SyncRuda);
        let memory = self.get_memory(op.list);
        let index = self.get_index(op.index, op.list.ty.with_vector_size(width), true);
        let value = if op.list.storage_type().is_atomic() {
            assert_eq!(width, 1, "CPU atomics require a scalar pointee");
            let pointer = self.memref_element_pointer(
                memory, index, out.storage_type().to_type(self.context),
            );
            let integer_type = IntegerType::new(self.context, 64).into();
            self.append_operation_with_result(
                OperationBuilder::new("llvm.load", self.location)
                    .add_operands(&[pointer])
                    .add_results(&[out.ty.to_type(self.context)])
                    .add_attributes(&[
                        (Identifier::new(self.context, "ordering"),
                            IntegerAttribute::new(integer_type, 2).into()),
                        (Identifier::new(self.context, "alignment"),
                            IntegerAttribute::new(integer_type, out.elem_type().size() as i64).into()),
                    ])
                    .build().expect("valid uniform atomic load"),
            )
        } else if width > 1 {
            self.append_operation_with_result(vector::load(
                self.context, out.ty.to_type(self.context), memory, &[index], self.location,
            ))
        } else {
            self.append_operation_with_result(memref::load(memory, &[index], self.location))
        };
        self.insert_variable(out, value);
    }

    pub(crate) fn visit_copy_memory_bulk(&mut self, op: &CopyMemoryBulkOperator, out: Variable) {
        assert_eq!(op.input.ty, out.ty, "memory copy requires matching element types and vector widths");
        if op.len == 0 {
            return;
        }
        let source = self.get_memory(op.input);
        let destination = self.get_memory(out);
        let in_index = self.get_index(op.in_index, op.input.ty, true);
        let in_offset = self.get_index(op.offset_input, op.input.ty, true);
        let in_start = self.append_operation_with_result(arith::addi(
            in_index, in_offset, self.location,
        ));
        let out_index = self.get_index(op.out_index, out.ty, true);
        let out_offset = self.get_index(op.offset_out, out.ty, true);
        let out_start = self.append_operation_with_result(arith::addi(
            out_index, out_offset, self.location,
        ));
        // CPU memrefs store scalar lanes, while the IR length and offsets count lines.
        let length = op.len.checked_mul(out.vector_size())
            .and_then(|length| i64::try_from(length).ok())
            .expect("memory copy length exceeds the CPU index range");
        let index_type = Type::index(self.context);
        let start = self.block.const_int_from_type(self.context, self.location, 0, index_type).unwrap();
        let end = self.block.const_int_from_type(self.context, self.location, length, index_type).unwrap();
        let step = self.block.const_int_from_type(self.context, self.location, 1, index_type).unwrap();
        let region = Region::new();
        let block = Block::new(&[(index_type, self.location)]);
        let index = block.argument(0).unwrap().into();
        let source_index = block.addi(in_start, index, self.location).unwrap();
        let destination_index = block.addi(out_start, index, self.location).unwrap();
        let value = block.append_op_result(memref::load(
            source, &[source_index], self.location,
        )).unwrap();
        block.append_operation(memref::store(
            value, destination, &[destination_index], self.location,
        ));
        block.append_operation(scf::r#yield(&[], self.location));
        region.append_block(block);
        self.block.append_operation(scf::r#for(start, end, step, region, self.location));
    }

    pub(crate) fn memref_element_pointer(
        &self,
        memory: Value<'a, 'a>,
        element_index: Value<'a, 'a>,
        element_type: Type<'a>,
    ) -> Value<'a, 'a> {
        let metadata = self.block.append_operation(
            OperationBuilder::new("memref.extract_strided_metadata", self.location)
                .add_operands(&[memory])
                .enable_result_type_inference()
                .build().expect("valid rank-one memref metadata"),
        );
        let offset = metadata.result(1).unwrap().into();
        let stride = metadata.result(3).unwrap().into();
        let scaled = self.append_operation_with_result(arith::muli(
            element_index, stride, self.location,
        ));
        let element_offset = self.append_operation_with_result(arith::addi(
            offset, scaled, self.location,
        ));
        let base = self.append_operation_with_result(
            OperationBuilder::new("memref.extract_aligned_pointer_as_index", self.location)
                .add_operands(&[memory])
                .add_results(&[Type::index(self.context)])
                .build().expect("valid memref base address"),
        );
        let integer_type = IntegerType::new(self.context, 64).into();
        let base = self.append_operation_with_result(index::castu(base, integer_type, self.location));
        let offset = self.append_operation_with_result(index::casts(
            element_offset, integer_type, self.location,
        ));
        let pointer = self.native_pointer(base);
        self.append_operation_with_result(llvm::get_element_ptr_dynamic(
            self.context, pointer, &[offset], element_type,
            llvm::r#type::pointer(self.context, 0), self.location,
        ))
    }

    pub(crate) fn native_pointer(&self, address: Value<'a, 'a>) -> Value<'a, 'a> {
        self.append_operation_with_result(
            OperationBuilder::new("llvm.inttoptr", self.location)
                .add_operands(&[address])
                .add_results(&[llvm::r#type::pointer(self.context, 0)])
                .build().expect("valid native pointer"),
        )
    }

    pub(crate) fn pointer_as_integer(
        &self, pointer: Value<'a, 'a>, ty: Type<'a>,
    ) -> Value<'a, 'a> {
        self.append_operation_with_result(
            OperationBuilder::new("llvm.ptrtoint", self.location)
                .add_operands(&[pointer])
                .add_results(&[ty])
                .build().expect("valid native address"),
        )
    }

    pub(crate) fn visit_native_address(&mut self, op: &BinaryOperator, out: Variable) {
        let memory = self.get_memory(op.lhs);
        let index = self.get_index(op.rhs, op.lhs.ty, op.lhs.ty.is_vectorized());
        let pointer = self.memref_element_pointer(
            memory, index, op.lhs.storage_type().to_type(self.context),
        );
        let value = self.pointer_as_integer(pointer, out.ty.to_type(self.context));
        self.insert_variable(out, value);
    }

    pub(crate) fn visit_native_load(&mut self, op: &UnaryOperator, out: Variable) {
        let pointer = self.native_pointer(self.get_variable(op.input));
        let value = self.append_operation_with_result(
            OperationBuilder::new("llvm.load", self.location)
                .add_operands(&[pointer])
                .add_results(&[out.ty.to_type(self.context)])
                .add_attributes(&[(
                    Identifier::new(self.context, "alignment"),
                    IntegerAttribute::new(IntegerType::new(self.context, 64).into(), 1).into(),
                )])
                .build().expect("valid native load"),
        );
        self.insert_variable(out, value);
    }

    pub(crate) fn visit_native_store(&mut self, op: &BinaryOperator) {
        let pointer = self.native_pointer(self.get_variable(op.lhs));
        let value = self.get_variable(op.rhs);
        self.block.append_operation(
            OperationBuilder::new("llvm.store", self.location)
                .add_operands(&[value, pointer])
                .add_attributes(&[(
                    Identifier::new(self.context, "alignment"),
                    IntegerAttribute::new(IntegerType::new(self.context, 64).into(), 1).into(),
                )])
                .build().expect("valid native store"),
        );
    }
}
