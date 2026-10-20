use std::collections::HashMap;

use ruda_core::ir::{self, Builtin, ConstantValue, FloatKind, VariableKind};
use melior::{
    dialect::{
        index, memref,
        ods::{arith, vector},
    },
    ir::{
        attribute::{FloatAttribute, IntegerAttribute},
        r#type::{IntegerType, MemRefType},
    },
};

use super::prelude::*;

#[derive(Default, Debug)]
pub struct Variables<'a> {
    pub local: HashMap<VariableKind, Value<'a, 'a>>,
    pub global_constant: HashMap<u32, (ir::Type, usize)>,
}

impl<'a> Variables<'a> {
    pub fn new(opt: &Optimizer) -> Self {
        let mut variables = Self::default();
        for const_array in opt.const_arrays() {
            variables
                .global_constant
                .insert(const_array.id, (const_array.item, const_array.length));
        }
        variables
    }
}

impl<'a> Visitor<'a> {
    pub fn insert_variable(&mut self, variable: Variable, value: Value<'a, 'a>) {
        match variable.kind {
            VariableKind::LocalConst { .. } | VariableKind::Versioned { .. } => {
                self.variables.local.insert(variable.kind, value);
            }
            VariableKind::LocalMut { .. } => {
                self.insert_mutable_memory(variable, value, 1);
            }
            VariableKind::LocalArray { length, .. } => {
                self.insert_mutable_memory(variable, value, length);
            }
            VariableKind::Shared { .. } => {
                self.insert_shared_memory(variable, value);
            }
            _ => todo!("This variable is not implemented {:?}", variable),
        };
    }

    fn get_mutable_memory(&mut self, variable: Variable, length: usize) -> Value<'a, 'a> {
        let r#type = if variable.storage_type().is_atomic()
            && !matches!(variable.kind, VariableKind::LocalArray { .. })
        {
            IntegerType::new(self.context, 64).into()
        } else {
            variable.storage_type().to_type(self.context)
        };
        let memref_type = MemRefType::new(
            r#type,
            &[variable.vector_size() as i64 * length as i64],
            None,
            None,
        );
        self.variables
            .local
            .get(&variable.kind)
            .copied()
            .unwrap_or_else(|| {
                let value = self
                    .first_block
                    .unwrap()
                    .append_op_result(memref::alloca(
                        self.context,
                        memref_type,
                        &[],
                        &[],
                        None,
                        self.location,
                    ))
                    .unwrap();
                self.variables.local.insert(variable.kind, value);
                value
            })
    }

    fn insert_mutable_memory(&mut self, variable: Variable, value: Value<'a, 'a>, length: usize) {
        let memref = self.get_mutable_memory(variable, length);
        let value = if variable.storage_type().is_atomic()
            && !matches!(variable.kind, VariableKind::LocalArray { .. })
        {
            self.pointer_as_integer(value, IntegerType::new(self.context, 64).into())
        } else {
            value
        };
        let integer = IntegerAttribute::new(Type::index(self.context), 0).into();
        let zero = self.append_operation_with_result(arith::constant(
            self.context,
            Type::index(self.context),
            integer,
            self.location,
        ));
        let operation = if value.r#type().is_vector() {
            vector::store(self.context, value, memref, &[zero], self.location).into()
        } else {
            memref::store(value, memref, &[zero], self.location)
        };
        self.block.append_operation(operation);
    }

    fn insert_shared_memory(&mut self, variable: Variable, value: Value<'a, 'a>) {
        let id = variable.index().unwrap();
        let memref = *self.args_manager.shared_memory_values.get(&id).unwrap();
        let integer = IntegerAttribute::new(Type::index(self.context), 0).into();
        let zero = self.append_operation_with_result(arith::constant(
            self.context,
            Type::index(self.context),
            integer,
            self.location,
        ));
        let operation = if value.r#type().is_vector() {
            vector::store(self.context, value, memref, &[zero], self.location).into()
        } else {
            memref::store(value, memref, &[zero], self.location)
        };
        self.block.append_operation(operation);
    }

    pub fn get_binary_op_variable(
        &self,
        lhs: Variable,
        rhs: Variable,
    ) -> (Value<'a, 'a>, Value<'a, 'a>) {
        let vectorization_factor = std::cmp::max(lhs.vector_size(), rhs.vector_size());
        let (mut lhs_value, mut rhs_value) = (self.get_variable(lhs), self.get_variable(rhs));

        if lhs_value.r#type().is_vector() || rhs_value.r#type().is_vector() {
            if !lhs_value.r#type().is_vector() {
                let vector_type = Type::vector(
                    &[vectorization_factor as u64],
                    lhs.storage_type().to_type(self.context),
                );
                lhs_value = self.append_operation_with_result(vector::splat(
                    self.context,
                    vector_type,
                    lhs_value,
                    self.location,
                ));
            }
            if !rhs_value.r#type().is_vector() {
                let vector_type = Type::vector(
                    &[vectorization_factor as u64],
                    rhs.storage_type().to_type(self.context),
                );
                rhs_value = self.append_operation_with_result(vector::splat(
                    self.context,
                    vector_type,
                    rhs_value,
                    self.location,
                ));
            }
        }
        (lhs_value, rhs_value)
    }

    pub fn get_memory(&mut self, variable: Variable) -> Value<'a, 'a> {
        match variable.kind {
            VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) => {
                self.args_manager.buffers[id as usize]
            }
            VariableKind::SharedArray { id, .. } => *self
                .args_manager
                .shared_memory_values
                .get(&id)
                .expect("Variable should have been declared before"),
            VariableKind::Shared { id, .. } => *self
                .args_manager
                .shared_memory_values
                .get(&id)
                .expect("Variable should have been declared before"),
            VariableKind::LocalMut { .. } | VariableKind::LocalConst { .. } => {
                self.get_mutable_memory(variable, 1)
            }
            VariableKind::LocalArray { length, .. } => self.get_mutable_memory(variable, length),
            VariableKind::ConstantArray { id, .. } => {
                let name = id.to_string();
                let (item, length) = self
                    .variables
                    .global_constant
                    .get(&id)
                    .copied()
                    .unwrap();
                let length = length.checked_mul(item.vector_size())
                    .and_then(|length| i64::try_from(length).ok())
                    .expect("constant array length exceeds the CPU index range");
                let r#type = MemRefType::new(
                    item.storage_type().to_type(self.context), &[length], None, None,
                );
                self.append_operation_with_result(memref::get_global(
                    self.context,
                    &name,
                    r#type,
                    self.location,
                ))
            }
            _ => todo!(
                "This variable isn't backed by memory or implemented: {}",
                variable
            ),
        }
    }

    pub fn is_memory(&self, variable: Variable) -> bool {
        matches!(
            variable.kind,
            VariableKind::GlobalInputArray(_)
                | VariableKind::GlobalOutputArray(_)
                | VariableKind::LocalMut { .. }
                | VariableKind::LocalArray { .. }
                | VariableKind::ConstantArray { .. }
                | VariableKind::SharedArray { .. }
                | VariableKind::Shared { .. }
        )
    }

    pub fn get_variable(&self, variable: Variable) -> Value<'a, 'a> {
        match variable.kind {
            VariableKind::LocalConst { .. } => *self
                .variables
                .local
                .get(&variable.kind)
                .expect("Variable should have been declared before"),
            VariableKind::Builtin(builtin) => self.get_builtin(builtin),
            VariableKind::Constant(constant_scalar_value) => {
                let (const_type, attribute) = match constant_scalar_value {
                    ConstantValue::Int(value) => {
                        let size = variable.ty.elem_type().size_bits() as u32;

                        let integer_type = IntegerType::new(self.context, size).into();
                        let integer_attribute = IntegerAttribute::new(integer_type, value).into();
                        (integer_type, integer_attribute)
                    }
                    ConstantValue::UInt(value) => {
                        let size = variable.ty.elem_type().size_bits() as u32;

                        let integer_type = IntegerType::new(self.context, size).into();
                        let integer_attribute =
                            IntegerAttribute::new(integer_type, value as i64).into();
                        (integer_type, integer_attribute)
                    }
                    ConstantValue::Float(value) => {
                        let float_type = match variable.ty.elem_type().as_float().unwrap() {
                            FloatKind::F16 => Type::float16(self.context),
                            FloatKind::BF16 => Type::bfloat16(self.context),
                            FloatKind::F32 => Type::float32(self.context),
                            FloatKind::F64 => Type::float64(self.context),
                            _ => panic!("Type is not supported in LLVM"),
                        };
                        let float_attribute =
                            FloatAttribute::new(self.context, float_type, value).into();
                        (float_type, float_attribute)
                    }
                    ConstantValue::Bool(bool) => {
                        let integer_type = IntegerType::new(self.context, 8).into();
                        let integer_attribute =
                            IntegerAttribute::new(integer_type, bool as i64).into();
                        (integer_type, integer_attribute)
                    }
                };
                let value = self.append_operation_with_result(arith::constant(
                    self.context,
                    const_type,
                    attribute,
                    self.location,
                ));
                match variable.ty.is_vectorized() {
                    true => {
                        let vector = Type::vector(&[variable.vector_size() as u64], const_type);
                        self.append_operation_with_result(vector::splat(
                            self.context,
                            vector,
                            value,
                            self.location,
                        ))
                    }
                    false => value,
                }
            }
            VariableKind::Versioned { .. } => *self
                .variables
                .local
                .get(&variable.kind)
                .expect("Variable should have been declared before"),
            VariableKind::LocalMut { .. } => {
                let memref = *self
                    .variables
                    .local
                    .get(&variable.kind)
                    .expect("Variable should have been declared before");
                let result_type = variable.ty.to_type(self.context);
                let integer = IntegerAttribute::new(Type::index(self.context), 0).into();
                let zero = self.append_operation_with_result(arith::constant(
                    self.context,
                    Type::index(self.context),
                    integer,
                    self.location,
                ));
                if variable.storage_type().is_atomic() {
                    let address = self.append_operation_with_result(memref::load(
                        memref, &[zero], self.location,
                    ));
                    self.native_pointer(address)
                } else if variable.ty.is_vectorized() {
                    self.append_operation_with_result(vector::load(
                        self.context,
                        result_type,
                        memref,
                        &[zero],
                        self.location,
                    ))
                } else {
                    self.append_operation_with_result(memref::load(memref, &[zero], self.location))
                }
            }
            VariableKind::GlobalScalar(id) => {
                let memref = *self
                    .args_manager
                    .scalars_memref
                    .get(&variable.storage_type())
                    .unwrap();
                let index = self
                    .block
                    .const_int_from_type(
                        self.context,
                        self.location,
                        id as i64,
                        Type::index(self.context),
                    )
                    .unwrap();
                let value = self.append_operation_with_result(memref::load(
                    memref,
                    &[index],
                    self.location,
                ));
                match variable.ty.is_vectorized() {
                    true => {
                        let vector = Type::vector(
                            &[variable.vector_size() as u64],
                            variable.storage_type().to_type(self.context),
                        );
                        self.append_operation_with_result(vector::splat(
                            self.context,
                            vector,
                            value,
                            self.location,
                        ))
                    }
                    false => value,
                }
            }
            VariableKind::Shared { id } => {
                let memref = *self.args_manager.shared_memory_values.get(&id).unwrap();
                let index = self
                    .block
                    .const_int_from_type(self.context, self.location, 0, Type::index(self.context))
                    .unwrap();
                if variable.storage_type().is_atomic() {
                    return self.memref_element_pointer(
                        memref, index, variable.storage_type().to_type(self.context),
                    );
                }
                let value = self.append_operation_with_result(memref::load(
                    memref,
                    &[index],
                    self.location,
                ));
                match variable.ty.is_vectorized() {
                    true => {
                        let vector = Type::vector(
                            &[variable.vector_size() as u64],
                            variable.storage_type().to_type(self.context),
                        );
                        self.append_operation_with_result(vector::splat(
                            self.context,
                            vector,
                            value,
                            self.location,
                        ))
                    }
                    false => value,
                }
            }
            _ => todo!("{:?} is not yet implemented", variable.kind),
        }
    }

    pub fn get_index(
        &self,
        variable: Variable,
        target_item: ir::Type,
        list_is_vectorized: bool,
    ) -> Value<'a, 'a> {
        let index = self.get_variable(variable);
        let cast = if variable.storage_type().is_unsigned_int() {
            index::castu(index, Type::index(self.context), self.location)
        } else {
            index::casts(index, Type::index(self.context), self.location)
        };
        let mut index = self.append_operation_with_result(cast);
        if target_item.is_vectorized() && list_is_vectorized {
            let vectorization = target_item.vector_size() as i64;
            let constant = self.append_operation_with_result(arith::constant(
                self.context,
                Type::index(self.context),
                IntegerAttribute::new(Type::index(self.context), vectorization).into(),
                self.location,
            ));
            index = self.append_operation_with_result(arith::muli(
                self.context,
                index,
                constant,
                self.location,
            ));
        }
        index
    }

    pub fn get_builtin(&self, builtin: Builtin) -> Value<'a, 'a> {
        self.args_manager.get(builtin)
    }
}
