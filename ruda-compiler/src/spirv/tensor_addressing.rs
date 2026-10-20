use ruda_core::ir::{self as core, TensorClampMode};
use rspirv::{
    dr::{InsertPoint, Instruction, Operand},
    spirv::{Capability, MemoryAccess, Op, TensorAddressingOperands, Word},
};

use super::{SpirvCompiler, SpirvTarget, item::Item, variable::ConstVal};

impl<T: SpirvTarget> SpirvCompiler<T> {
    fn enable_tensor_addressing(&mut self) {
        assert!(self.compilation_options.vulkan.max_spirv_version >= (1, 6),
            "Tensor addressing requires SPIR-V 1.6");
        self.capabilities.insert(Capability::TensorAddressingNV);
    }

    pub(super) fn tensor_layout_type(&mut self, mut layout: core::TensorLayoutType) -> Word {
        self.enable_tensor_addressing();
        assert!((1..=5).contains(&layout.rank), "Invalid tensor layout rank");
        let clamp_mode = match layout.clamp_mode {
            TensorClampMode::Undefined => 0,
            TensorClampMode::Constant(_) => {
                layout.clamp_mode = TensorClampMode::Constant(0);
                1
            }
            TensorClampMode::ClampToEdge => 2,
            TensorClampMode::Repeat => 3,
            TensorClampMode::RepeatMirrored => 4,
        };
        let key = Item::TensorLayout(layout);
        if let Some(id) = self.state.tensor_types.get(&key) {
            return *id;
        }
        let rank = self.const_u32(layout.rank);
        let clamp = self.const_u32(clamp_mode);
        let id = self.id();
        self.module_mut().types_global_values.push(Instruction::new(
            Op::TypeTensorLayoutNV, None, Some(id),
            vec![Operand::IdRef(rank), Operand::IdRef(clamp)],
        ));
        self.state.tensor_types.insert(key, id);
        id
    }

    pub(super) fn tensor_view_type(&mut self, mut view: core::TensorViewType) -> Word {
        self.enable_tensor_addressing();
        assert!((1..=5).contains(&view.rank), "Invalid tensor view rank");
        let mut used = [false; 5];
        for axis in &view.permutation[..view.rank as usize] {
            assert!(*axis < view.rank && !used[*axis as usize], "Invalid tensor permutation");
            used[*axis as usize] = true;
        }
        view.permutation[view.rank as usize..].fill(0);
        let key = Item::TensorView(view);
        if let Some(id) = self.state.tensor_types.get(&key) {
            return *id;
        }
        let rank = self.const_u32(view.rank);
        let boolean = self.type_bool();
        let has_dimensions = self.constant_false(boolean);
        let mut operands = vec![Operand::IdRef(rank), Operand::IdRef(has_dimensions)];
        for axis in &view.permutation[..view.rank as usize] {
            operands.push(Operand::IdRef(self.const_u32(*axis)));
        }
        let id = self.id();
        self.module_mut().types_global_values.push(Instruction::new(
            Op::TypeTensorViewNV, None, Some(id), operands,
        ));
        self.state.tensor_types.insert(key, id);
        id
    }

    fn tensor_instruction(&mut self, op: Op, ty: Word, result: Word, operands: Vec<Operand>) {
        self.insert_into_block(InsertPoint::End, Instruction::new(
            op, Some(ty), Some(result), operands,
        )).unwrap();
    }

    fn tensor_value(&mut self, value: core::Variable) -> Word {
        let value = self.compile_variable(value);
        self.read(&value)
    }

    pub(super) fn compile_tensor_layout(
        &mut self,
        out: core::Variable,
        shape: Vec<core::Variable>,
        strides: Option<Vec<core::Variable>>,
    ) {
        let layout = match out.ty {
            core::Type::Semantic(core::SemanticType::TensorLayout(layout)) => layout,
            _ => panic!("Tensor layout output must have a tensor layout type"),
        };
        assert_eq!(shape.len(), layout.rank as usize);
        let clamp_value = match layout.clamp_mode {
            TensorClampMode::Constant(value) if value != 0 => Some(value),
            _ => None,
        };
        let out = self.compile_variable(out);
        let ty = out.item().id(self);
        let out_id = self.write_id(&out);
        let mut value = self.id();
        self.tensor_instruction(Op::CreateTensorLayoutNV, ty, value, vec![]);
        let mut operands = vec![Operand::IdRef(value)];
        operands.extend(shape.into_iter().map(|dim| Operand::IdRef(self.tensor_value(dim))));
        value = if strides.is_none() && clamp_value.is_none() { out_id } else { self.id() };
        self.tensor_instruction(Op::TensorLayoutSetDimensionNV, ty, value, operands);
        if let Some(strides) = strides {
            assert_eq!(strides.len(), layout.rank as usize);
            let mut operands = vec![Operand::IdRef(value)];
            operands.extend(strides.into_iter().map(|stride| Operand::IdRef(self.tensor_value(stride))));
            value = if clamp_value.is_none() { out_id } else { self.id() };
            self.tensor_instruction(Op::TensorLayoutSetStrideNV, ty, value, operands);
        }
        if let Some(clamp_value) = clamp_value {
            let clamp_value = self.const_u32(clamp_value);
            self.tensor_instruction(Op::TensorLayoutSetClampValueNV, ty, out_id,
                vec![Operand::IdRef(value), Operand::IdRef(clamp_value)]);
        }
        self.write(&out, out_id);
    }

    pub(super) fn compile_tensor_view(&mut self, out: core::Variable) {
        let out = self.compile_variable(out);
        let ty = out.item().id(self);
        let out_id = self.write_id(&out);
        self.tensor_instruction(Op::CreateTensorViewNV, ty, out_id, vec![]);
        self.write(&out, out_id);
    }

    pub(super) fn compile_tensor_slice(
        &mut self, out: core::Variable, layout: core::Variable,
        offsets: Vec<core::Variable>, shape: Vec<core::Variable>,
    ) {
        let rank = match layout.ty {
            core::Type::Semantic(core::SemanticType::TensorLayout(layout)) => layout.rank as usize,
            _ => panic!("Tensor slice requires a layout"),
        };
        assert_eq!(out.ty, layout.ty);
        assert_eq!(offsets.len(), rank);
        assert_eq!(shape.len(), rank);
        let mut operands = vec![Operand::IdRef(self.tensor_value(layout))];
        for (offset, span) in offsets.into_iter().zip(shape) {
            operands.push(Operand::IdRef(self.tensor_value(offset)));
            operands.push(Operand::IdRef(self.tensor_value(span)));
        }
        let out = self.compile_variable(out);
        let ty = out.item().id(self);
        let out_id = self.write_id(&out);
        self.tensor_instruction(Op::TensorLayoutSliceNV, ty, out_id, operands);
        self.write(&out, out_id);
    }

    fn tensor_memory_operands(&mut self, alignment: u32, view: Option<core::Variable>) -> Vec<Operand> {
        let mut operands = vec![
            Operand::MemoryAccess(MemoryAccess::ALIGNED),
            Operand::LiteralBit32(alignment),
            Operand::TensorAddressingOperands(if view.is_some() {
                TensorAddressingOperands::TENSOR_VIEW
            } else {
                TensorAddressingOperands::NONE
            }),
        ];
        if let Some(view) = view {
            operands.push(Operand::IdRef(self.tensor_value(view)));
        }
        operands
    }

    pub(super) fn compile_load_tensor(
        &mut self, mat: core::Variable, buffer: core::Variable, offset: core::Variable,
        layout: core::Variable, view: Option<core::Variable>,
    ) {
        self.enable_tensor_addressing();
        self.capabilities.insert(Capability::CooperativeMatrixTensorAddressingNV);
        let mat = self.compile_variable(mat);
        let mat = self.matrix_var(&mat).1;
        let ty = self.item(&mat).id(self);
        let buffer = self.compile_variable(buffer);
        let offset = self.compile_variable(offset);
        let ptr = self.index_ptr(&buffer, &offset);
        let zero = if mat.elem.size() == 8 { ConstVal::Bit64(0) } else { ConstVal::Bit32(0) };
        let zero = mat.elem.constant(self, zero);
        let zero = self.constant_composite(ty, vec![zero]);
        let layout = self.tensor_value(layout);
        let mut operands = vec![Operand::IdRef(ptr), Operand::IdRef(zero), Operand::IdRef(layout)];
        operands.extend(self.tensor_memory_operands(mat.elem.size(), view));
        let result = self.id();
        self.tensor_instruction(Op::CooperativeMatrixLoadTensorNV, ty, result, operands);
        self.store(mat.id, result, None, vec![]).unwrap();
    }

    pub(super) fn compile_store_tensor(
        &mut self, mat: core::Variable, buffer: core::Variable, offset: core::Variable,
        layout: core::Variable, view: Option<core::Variable>,
    ) {
        self.enable_tensor_addressing();
        self.capabilities.insert(Capability::CooperativeMatrixTensorAddressingNV);
        let mat = self.compile_variable(mat);
        let mat = self.matrix_var(&mat).1;
        let ty = self.item(&mat).id(self);
        let matrix = self.load(ty, None, mat.id, None, vec![]).unwrap();
        let buffer = self.compile_variable(buffer);
        let offset = self.compile_variable(offset);
        let ptr = self.index_ptr(&buffer, &offset);
        let layout = self.tensor_value(layout);
        let mut operands = vec![Operand::IdRef(ptr), Operand::IdRef(matrix), Operand::IdRef(layout)];
        operands.extend(self.tensor_memory_operands(mat.elem.size(), view));
        self.insert_into_block(InsertPoint::End, Instruction::new(
            Op::CooperativeMatrixStoreTensorNV, None, None, operands,
        )).unwrap();
    }
}
