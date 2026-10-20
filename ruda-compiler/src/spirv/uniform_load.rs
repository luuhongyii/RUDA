use ruda_core::ir::{IndexOperator, Synchronization, Variable as IrVariable, VariableKind};
use rspirv::spirv::{Capability, MemorySemantics, Scope, StorageClass};

use super::{SpirvCompiler, SpirvTarget, item::{Elem, Item}, variable::Variable};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub(super) fn compile_workgroup_uniform_load(
        &mut self, op: IndexOperator, output: IrVariable, uniform: bool,
    ) {
        assert!(matches!(op.list.kind, VariableKind::SharedArray { .. } | VariableKind::Shared { .. }));
        let width = if op.vector_size == 0 { op.list.vector_size() } else { op.vector_size };
        assert_eq!(output.vector_size(), width);
        assert_eq!(output.elem_type(), op.list.elem_type());
        let atomic = op.list.storage_type().is_atomic();
        let shared_value = matches!(op.list.kind, VariableKind::Shared { .. });
        let stored_width = op.list.vector_size();
        let input = self.compile_variable(op.list);
        let index = self.compile_variable(op.index);
        let out = self.compile_variable(output);
        let index_item = index.item();
        let index_type = index_item.id(self);
        let index = self.read(&index);
        let scalar_item = Item::Scalar(out.item().elem());
        let scalar_type = scalar_item.id(self);
        if atomic && matches!(scalar_item.elem(), Elem::Int(64, _)) {
            self.capabilities.insert(Capability::Int64Atomics);
        }
        let pointer_type = Item::Pointer(StorageClass::Workgroup, Box::new(scalar_item)).id(self);
        let width_constant = index_item.const_u32(self, u32::try_from(width).expect("vector width fits U32"));
        let stored_width_constant = index_item.const_u32(self, u32::try_from(stored_width).expect("vector width fits U32"));
        let start = self.i_mul(index_type, None, index, width_constant).unwrap();
        self.compile_sync(Synchronization::SyncRuda);
        let memory_scope = self.const_u32(Scope::Workgroup as u32);
        let semantics = self.const_u32((MemorySemantics::WORKGROUP_MEMORY | MemorySemantics::ACQUIRE).bits());
        let mut values = Vec::with_capacity(width);
        for lane in 0..width {
            let lane = index_item.const_u32(self, u32::try_from(lane).expect("vector lane fits U32"));
            let scalar_index = self.i_add(index_type, None, start, lane).unwrap();
            let pointer = if shared_value {
                if stored_width == 1 && !self.compilation_options.vulkan.supports_explicit_smem {
                    input.id(self)
                } else {
                    self.index_ptr(&input, &Variable::Raw(scalar_index, index_item.clone()))
                }
            } else {
                let array_index = if stored_width == 1 {
                    scalar_index
                } else {
                    self.u_div(index_type, None, scalar_index, stored_width_constant).unwrap()
                };
                let array_pointer = self.index_ptr(&input, &Variable::Raw(array_index, index_item.clone()));
                if stored_width == 1 {
                    array_pointer
                } else {
                    let component = self.u_mod(index_type, None, scalar_index, stored_width_constant).unwrap();
                    self.access_chain(pointer_type, None, array_pointer, vec![component]).unwrap()
                }
            };
            let value = if atomic {
                self.atomic_load(scalar_type, None, pointer, memory_scope, semantics).unwrap()
            } else {
                self.load(scalar_type, None, pointer, None, vec![]).unwrap()
            };
            self.mark_uniformity(value, uniform);
            values.push(value);
        }
        let out_type = out.item().id(self);
        let out_id = self.write_id(&out);
        if width == 1 {
            self.copy_object(out_type, Some(out_id), values[0]).unwrap();
        } else {
            self.composite_construct(out_type, Some(out_id), values).unwrap();
        }
        self.mark_uniformity(out_id, uniform);
        self.write(&out, out_id);
    }
}
