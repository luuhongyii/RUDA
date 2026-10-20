use alloc::{format, string::String, vec::Vec};

use super::{
    Branch, Instruction, NonSemantic, Operation, OperationReflect, Scope, Type, TypeHash, Variable,
    VariableKind,
};

/// A device function whose first parameters are explicit arguments and whose remaining
/// parameters bind the lexical captures of its source closure.
#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeHash)]
pub struct ScalarFunction {
    pub body: Scope,
    pub return_type: Type,
    pub num_arguments: usize,
}

impl ScalarFunction {
    /// Close a scalar callback and return the arguments to pass at the call site.
    pub fn capture(
        mut body: Scope,
        arguments: Vec<Variable>,
        result: Variable,
        first_local_id: u32,
    ) -> Result<(Self, Vec<Variable>), String> {
        if result.ty.is_semantic() || result.ty.is_pointer() || result.ty.vector_size() != 1 {
            return Err("a matrix element callback must return a scalar".into());
        }
        let mut captures = Vec::new();
        let mut collect = |var: Variable| {
            let external = match var.kind {
                VariableKind::Constant(_) => false,
                VariableKind::LocalMut { id }
                | VariableKind::LocalConst { id }
                | VariableKind::Versioned { id, .. }
                | VariableKind::LocalArray { id, .. }
                | VariableKind::ConstantArray { id, .. }
                | VariableKind::Matrix { id, .. }
                | VariableKind::Pipeline { id, .. }
                | VariableKind::BarrierToken { id, .. } => id < first_local_id,
                _ => true,
            };
            if external && !arguments.contains(&var) && !captures.contains(&var) {
                captures.push(var);
            }
        };
        Self::visit_body(&body, &mut collect)?;
        collect(result);
        let num_arguments = arguments.len();
        body.parameters = arguments;
        body.parameters.extend(captures.iter().copied());
        body.register(Instruction::no_out(Branch::ReturnValue(result)));
        Ok((Self { body, return_type: result.ty, num_arguments }, captures))
    }

    /// Matrix callbacks cannot execute collective/tangled instructions.
    /// Inlined scalar calls, branches, loops and memory operations retain their usual IR.
    fn visit_body(
        scope: &Scope,
        visit: &mut impl FnMut(Variable),
    ) -> Result<(), String> {
        for (_, values) in &scope.const_arrays {
            for value in values {
                visit(*value);
            }
        }
        for inst in &scope.instructions {
            if let Some(out) = inst.out {
                visit(out);
            }
            match &inst.operation {
                Operation::Branch(branch) => match branch {
                    Branch::If(b) => {
                        visit(b.cond);
                        Self::visit_body(&b.scope, visit)?;
                    }
                    Branch::IfElse(b) => {
                        visit(b.cond);
                        Self::visit_body(&b.scope_if, visit)?;
                        Self::visit_body(&b.scope_else, visit)?;
                    }
                    Branch::Switch(b) => {
                        visit(b.value);
                        Self::visit_body(&b.scope_default, visit)?;
                        for (value, scope) in &b.cases {
                            visit(*value);
                            Self::visit_body(scope, visit)?;
                        }
                    }
                    Branch::RangeLoop(b) => {
                        visit(b.i);
                        visit(b.start);
                        visit(b.end);
                        if let Some(step) = b.step {
                            visit(step);
                        }
                        Self::visit_body(&b.scope, visit)?;
                    }
                    Branch::Loop(b) => Self::visit_body(&b.scope, visit)?,
                    Branch::Return => return Err("a scalar callback cannot return without a value".into()),
                    Branch::ReturnValue(value) => visit(*value),
                    Branch::Break | Branch::Unreachable => {}
                },
                Operation::Plane(_) | Operation::Synchronization(_) | Operation::CoopMma(_)
                | Operation::Barrier(_) | Operation::Tma(_) => {
                    return Err(format!("collective operation is not allowed in a matrix callback: {}", inst.operation));
                }
                Operation::NonSemantic(NonSemantic::Print { args, .. }) => {
                    for var in args {
                        visit(*var);
                    }
                }
                Operation::NonSemantic(_) | Operation::Marker(_) => {}
                operation => {
                    for var in operation.args().ok_or_else(|| {
                        format!("cannot capture arguments of callback operation: {operation}")
                    })? {
                        visit(var);
                    }
                }
            }
        }
        Ok(())
    }
}
