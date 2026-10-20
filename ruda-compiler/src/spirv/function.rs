use std::{mem, rc::Rc};

use ruda_core::{ir, kernel::KernelDefinition};
use rspirv::{
    dr::{InsertPoint, Instruction, Operand},
    spirv::{Capability, Op, StorageClass, Word},
};

use crate::optimizer::Uniformity;
use super::{SpirvCompiler, SpirvTarget, item::Item, variable::Variable};

#[derive(Clone)]
enum ParameterBinding {
    Value,
    LocalArray { item: Item, len: u32, constant: bool },
    Global(Variable),
}

#[derive(Clone)]
struct FunctionParameter {
    formal: ir::Variable,
    ty: Option<Word>,
    binding: ParameterBinding,
}

#[derive(Clone)]
pub(super) struct PendingFunction {
    id: Word,
    function: ir::ScalarFunction,
    parameters: Vec<FunctionParameter>,
}

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub(super) fn compile_matrix_elementwise(
        &mut self,
        input: ir::Variable,
        output: ir::Variable,
        function: ir::ScalarFunction,
        captures: Vec<ir::Variable>,
    ) {
        assert!(self.compilation_options.vulkan.max_spirv_version >= (1, 6),
            "cooperative matrix callbacks require SPIR-V 1.6");
        self.capabilities.insert(Capability::CooperativeMatrixPerElementOperationsNV);
        let input = self.compile_variable(input);
        let output = self.compile_variable(output);
        let input = self.matrix_var(&input).1;
        let output = self.matrix_var(&output).1;
        let item = self.item(&input);
        let ty = item.id(self);
        let output_ty = self.item(&output).id(self);
        assert_eq!(ty, output_ty, "elementwise input and output matrix types must match");
        let return_ty = self.compile_type(function.return_type).id(self);
        assert_eq!(return_ty, input.elem.id(self),
            "callback return type must match the matrix component type");
        assert_eq!(function.num_arguments, 3);
        assert_eq!(function.body.parameters.len(), 3 + captures.len());
        let mut parameters = Vec::new();
        for formal in &function.body.parameters[..3] {
            let ty = self.compile_type(formal.ty).id(self);
            parameters.push(FunctionParameter {
                formal: *formal, ty: Some(ty), binding: ParameterBinding::Value,
            });
        }
        let matrix = self.load(ty, None, input.id, None, []).unwrap();
        let function_id = self.id();
        let mut operands = vec![Operand::IdRef(matrix), Operand::IdRef(function_id)];
        for (formal, actual) in function.body.parameters[3..].iter().copied().zip(captures) {
            let array_factor = match actual.kind {
                ir::VariableKind::LocalArray { unroll_factor, .. }
                | ir::VariableKind::ConstantArray { unroll_factor, .. } => unroll_factor as u32,
                _ => 1,
            };
            let actual = self.compile_variable(actual);
            let (ty, binding) = match &actual {
                // Shader-global storage remains the same storage in every helper.
                Variable::GlobalInputArray(..) | Variable::GlobalOutputArray(..)
                | Variable::SharedArray(..) | Variable::Shared(..) => {
                    (None, ParameterBinding::Global(actual.clone()))
                }
                Variable::LocalArray(id, item, len) | Variable::ConstantArray(id, item, len) => {
                    assert!(self.compilation_options.vulkan.supports_variable_pointers,
                        "capturing function-local array storage requires Vulkan variablePointers");
                    self.capabilities.insert(Capability::VariablePointers);
                    let ty = Item::Pointer(StorageClass::Function,
                        Box::new(Item::Array(Box::new(item.clone()), *len * array_factor))).id(self);
                    operands.push(Operand::IdRef(*id));
                    (Some(ty), ParameterBinding::LocalArray {
                        item: item.clone(), len: *len,
                        constant: matches!(actual, Variable::ConstantArray(..)),
                    })
                }
                _ => {
                    let item = self.compile_type(formal.ty);
                    let ty = item.id(self);
                    let value = self.read_as(&actual, &item);
                    operands.push(Operand::IdRef(value));
                    (Some(ty), ParameterBinding::Value)
                }
            };
            parameters.push(FunctionParameter { formal, ty, binding });
        }
        let result = self.id();
        self.insert_into_block(InsertPoint::End, Instruction::new(
            Op::CooperativeMatrixPerElementOpNV, Some(ty), Some(result), operands,
        )).unwrap();
        self.store(output.id, result, None, []).unwrap();
        self.pending_functions.push(PendingFunction { id: function_id, function, parameters });
    }

    /// Emit callbacks after the entry function, with isolated function-local lookup tables.
    pub(super) fn compile_functions(&mut self, kernel: &KernelDefinition) {
        let pending = mem::take(&mut self.pending_functions);
        if pending.is_empty() {
            return;
        }
        let caller_index = self.module_ref().functions.len() - 1;
        let main_opt = self.opt.clone();
        let main_uniformity = self.uniformity.clone();
        let main_debug = self.debug_info.clone();
        let main_setup = self.setup_block;
        let main_visited = mem::take(&mut self.visited);
        let main_block = self.current_block.take();
        for callback in pending {
            let mut saved_state = self.state.clone();
            self.state.bindings.clear();
            self.state.versioned.clear();
            self.state.variables.clear();
            self.state.local_arrays.clear();
            self.state.const_arrays.clear();
            self.state.matrices.clear();
            self.state.labels.clear();
            self.state.end_labels.clear();
            self.state.loops.clear();
            self.state.slices.clear();
            self.state.function_values.clear();
            self.state.globals.clear();
            self.state.loaded_builtins.clear();
            self.state.scalars.clear();
            self.visited.clear();
            self.current_block = None;

            let mut definition = kernel.clone();
            definition.body = callback.function.body;
            let mut opt = (self.lowering.optimize)(&definition, self.mode, &self.compilation_options);
            self.uniformity = opt.analysis::<Uniformity>();
            self.opt = Rc::new(opt);
            let return_ty = self.compile_type(callback.function.return_type).id(self);
            let types = callback.parameters.iter().filter_map(|param| param.ty).collect::<Vec<_>>();
            let function_ty = self.type_function(return_ty, types.clone());
            self.init_function_debug(return_ty, &types);
            let (_, debug_setup) = self.declare_device_function(
                &format!("matrix_elementwise_{}", callback.id),
                Some(callback.id), return_ty, function_ty,
            );
            let mut mutable = Vec::new();
            for parameter in callback.parameters {
                let value = parameter.ty.map(|ty| self.function_parameter(ty).unwrap());
                match parameter.binding {
                    ParameterBinding::Global(value) => {
                        self.state.function_values.insert(parameter.formal, value);
                    }
                    ParameterBinding::LocalArray { item, len, constant } => {
                        let id = value.unwrap();
                        let value = if constant { Variable::ConstantArray(id, item, len) }
                            else { Variable::LocalArray(id, item, len) };
                        self.state.function_values.insert(parameter.formal, value);
                    }
                    ParameterBinding::Value => {
                        let value = value.unwrap();
                        match parameter.formal.kind {
                            ir::VariableKind::LocalConst { id } => {
                                self.state.bindings.insert(id, value);
                                self.state.versioned.insert((id, 0), value);
                            }
                            ir::VariableKind::LocalMut { id } => {
                                self.state.versioned.insert((id, 0), value);
                                mutable.push((parameter.formal, value));
                            }
                            ir::VariableKind::Versioned { id, version } => {
                                self.state.versioned.insert((id, version), value);
                            }
                            _ => {
                                let item = self.compile_type(parameter.formal.ty);
                                self.state.function_values.insert(parameter.formal, Variable::Raw(value, item));
                            }
                        }
                    }
                }
            }
            let setup = self.id();
            self.setup_block = self.setup(setup, debug_setup);
            self.select_block(Some(self.setup_block)).unwrap();
            for (formal, value) in mutable {
                let local = self.compile_variable(formal);
                let ptr = local.id(self);
                self.store(ptr, value, None, []).unwrap();
            }
            let entry = self.opt.entry();
            let body = self.label(entry);
            self.select_block(None).unwrap();
            self.compile_block(entry);
            let ret = self.opt.ret;
            self.compile_block(ret);
            if self.selected_block().is_some() {
                let label = self.label(ret);
                self.branch(label).unwrap();
            }
            self.select_block(Some(self.setup_block)).unwrap();
            self.branch(body).unwrap();
            self.end_function().unwrap();

            // These entries refer to module-scope declarations, not function-local SSA ids.
            saved_state.used_builtins.extend(self.state.used_builtins.clone());
            saved_state.constants.extend(self.state.constants.clone());
            saved_state.tensor_types.extend(self.state.tensor_types.clone());
            saved_state.decorated_types.extend(self.state.decorated_types.iter().copied());
            saved_state.debug_types.extend(self.state.debug_types.iter().copied());
            saved_state.array_types.extend(self.state.array_types.iter().copied());
            self.state = saved_state;
        }
        assert!(self.pending_functions.is_empty(), "matrix callbacks cannot contain matrix operations");
        self.module_mut().functions[caller_index..].rotate_left(1);
        self.opt = main_opt;
        self.uniformity = main_uniformity;
        self.debug_info = main_debug;
        self.setup_block = main_setup;
        self.visited = main_visited;
        self.current_block = main_block;
    }
}
