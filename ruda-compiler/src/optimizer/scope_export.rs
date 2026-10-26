use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use ruda_core::{ir::*, launch::RudaDim};

use super::{ControlFlow, NodeIndex, Optimizer};

/// Run typed-pointer SSA optimization, then lower at a structured GPU backend boundary.
/// Kernels without pointer values retain their existing compilation path.
pub fn lower_optimized_pointer_values(scope: &mut Scope, dim: RudaDim) -> Result<(), String> {
    scope.lower_reference_values()?;
    fn present(scope: &Scope) -> bool {
        scope.instructions.iter().any(|inst| {
            matches!(inst.operation, Operation::Pointer(_))
                || inst.out.is_some_and(|out| out.ty.is_pointer())
                || match &inst.operation {
                    Operation::CoopMma(CoopMma::Elementwise { function, captures, .. }) => {
                        captures.iter().any(|capture| capture.ty.is_pointer()) || present(&function.body)
                    }
                    Operation::Branch(branch) => match branch {
                        Branch::If(b) => present(&b.scope),
                        Branch::IfElse(b) => present(&b.scope_if) || present(&b.scope_else),
                        Branch::Switch(b) => {
                            present(&b.scope_default) || b.cases.iter().any(|(_, s)| present(s))
                        }
                        Branch::RangeLoop(b) => present(&b.scope),
                        Branch::Loop(b) => present(&b.scope),
                        _ => false,
                    },
                    _ => false,
                }
        })
    }
    if !present(scope) {
        return Ok(());
    }
    PointerValueLowering::new(scope)?;
    let mut optimizer = Optimizer::new_pointer_values(scope.clone(), dim, vec![], vec![]);
    optimizer.lower_pointer_values()?;
    *scope = optimizer.into_scope()?;
    scope.lower_pointer_values()
}

#[derive(Clone, Copy)]
struct LoopRegion {
    header: NodeIndex,
    merge: NodeIndex,
}

impl Optimizer {
    /// Export structured SSA as scopes, placing parallel phi copies on their incoming edges.
    pub fn into_scope(mut self) -> Result<Scope, String> {
        let mut scope = self.root_scope.child();
        let variables = Rc::new(RefCell::new(HashSet::new()));
        let collect = {
            let variables = variables.clone();
            move |_: &mut Self, var: &mut Variable| {
                variables.borrow_mut().insert(*var);
            }
        };
        self.visit_all(collect.clone(), collect);
        let mut replacements = HashMap::new();
        let mut variables = variables.borrow().iter().copied().collect::<Vec<_>>();
        variables.sort_by_key(|var| format!("{var:?}"));
        for var in variables {
            match var.kind {
                VariableKind::LocalMut { .. }
                | VariableKind::LocalConst { .. }
                | VariableKind::Versioned { .. } => {
                    let replacement = *scope.create_local_restricted(var.ty);
                    scope.add_local_mut(replacement);
                    replacements.insert(var, replacement);
                }
                VariableKind::Matrix { .. } => scope.add_matrix(var),
                _ => {}
            }
        }
        let replace = |_: &mut Self, var: &mut Variable| {
            if let Some(replacement) = replacements.get(var) {
                *var = *replacement;
            }
        };
        self.visit_all(replace, replace);
        scope.parameters = self.root_scope.parameters.iter().map(|parameter| {
            let initial = match parameter.kind {
                VariableKind::LocalMut { id } | VariableKind::LocalConst { id } => {
                    Variable::new(VariableKind::Versioned { id, version: 0 }, parameter.ty)
                }
                _ => *parameter,
            };
            replacements.get(&initial).or_else(|| replacements.get(parameter))
                .copied().unwrap_or(*parameter)
        }).collect();
        for array in &self.program.const_arrays {
            let var = Variable::new(
                VariableKind::ConstantArray {
                    id: array.id,
                    length: array.length,
                    unroll_factor: 1,
                },
                array.item,
            );
            scope.const_arrays.push((var, array.values.clone()));
        }
        self.export_region(&mut scope, self.entry(), None, None)?;
        Ok(scope)
    }

    fn export_edge(
        &self,
        scope: &mut Scope,
        source: NodeIndex,
        target: NodeIndex,
    ) -> Result<(), String> {
        let mut copies = Vec::new();
        for phi in self.program[target].phi_nodes.borrow().iter() {
            let value = phi
                .entries
                .iter()
                .find(|entry| entry.block == source)
                .ok_or_else(|| format!("missing phi edge {source:?} -> {target:?}"))?
                .value;
            if value != phi.out {
                let temp = *scope.create_local_restricted(value.ty);
                scope.add_local_mut(temp);
                scope.register(Instruction::new(Operation::Copy(value), temp));
                copies.push((temp, phi.out));
            }
        }
        for (value, out) in copies {
            scope.register(Instruction::new(Operation::Copy(value), out));
        }
        Ok(())
    }

    fn export_arm(
        &self,
        parent: &mut Scope,
        source: NodeIndex,
        target: NodeIndex,
        stop: Option<NodeIndex>,
        loop_region: Option<LoopRegion>,
    ) -> Result<(Scope, bool), String> {
        let mut scope = parent.child();
        self.export_edge(&mut scope, source, target)?;
        let falls_through = self.export_region(&mut scope, target, stop, loop_region)?;
        Ok((scope, falls_through))
    }

    fn export_region(
        &self,
        scope: &mut Scope,
        mut node: NodeIndex,
        stop: Option<NodeIndex>,
        loop_region: Option<LoopRegion>,
    ) -> Result<bool, String> {
        let mut visited = HashSet::new();
        loop {
            if let Some(region) = loop_region {
                if node == region.merge {
                    scope.register(Branch::Break);
                    return Ok(false);
                }
                if node == region.header {
                    return Ok(false);
                }
            }
            if Some(node) == stop {
                return Ok(true);
            }
            if !visited.insert(node) {
                return Err("cannot export unstructured control-flow cycle".into());
            }
            let block = self.block(node);
            let control = block.control_flow.borrow().clone();
            if let ControlFlow::Loop { body, merge, .. }
            | ControlFlow::LoopBreak { body, merge, .. } = control
            {
                let mut body_scope = scope.child();
                body_scope
                    .instructions
                    .extend(block.ops.borrow().values().cloned());
                let region = Some(LoopRegion {
                    header: node,
                    merge,
                });
                if let ControlFlow::LoopBreak { break_cond, .. } = control {
                    let (scope_if, _) =
                        self.export_arm(&mut body_scope, node, body, None, region)?;
                    let mut scope_else = body_scope.child();
                    self.export_edge(&mut scope_else, node, merge)?;
                    scope_else.register(Branch::Break);
                    body_scope.register(Branch::IfElse(Box::new(IfElse {
                        cond: break_cond,
                        scope_if,
                        scope_else,
                    })));
                } else {
                    self.export_edge(&mut body_scope, node, body)?;
                    self.export_region(&mut body_scope, body, None, region)?;
                }
                scope.register(Branch::Loop(Box::new(Loop { scope: body_scope })));
                node = merge;
                continue;
            }
            scope
                .instructions
                .extend(block.ops.borrow().values().cloned());
            match control {
                ControlFlow::None => {
                    let successors = self.successors(node);
                    if successors.len() != 1 {
                        return Err("invalid fallthrough during scope export".into());
                    }
                    self.export_edge(scope, node, successors[0])?;
                    node = successors[0];
                }
                ControlFlow::IfElse {
                    cond,
                    then,
                    or_else,
                    ..
                } => {
                    let join = block.scope_join;
                    let (scope_if, then_continues) =
                        self.export_arm(scope, node, then, join, loop_region)?;
                    let (scope_else, else_continues) =
                        self.export_arm(scope, node, or_else, join, loop_region)?;
                    scope.register(Branch::IfElse(Box::new(IfElse {
                        cond,
                        scope_if,
                        scope_else,
                    })));
                    if !then_continues && !else_continues {
                        return Ok(false);
                    }
                    node = join.ok_or("missing lexical if join")?;
                }
                ControlFlow::Switch {
                    value,
                    default,
                    branches,
                    ..
                } => {
                    let join = block.scope_join;
                    let (scope_default, mut continues) =
                        self.export_arm(scope, node, default, join, loop_region)?;
                    let mut cases = Vec::new();
                    for (value, target) in branches {
                        let (branch, branch_continues) =
                            self.export_arm(scope, node, target, join, loop_region)?;
                        continues |= branch_continues;
                        cases.push((Variable::from(value), branch));
                    }
                    scope.register(Branch::Switch(Box::new(Switch {
                        value,
                        scope_default,
                        cases,
                    })));
                    if !continues {
                        return Ok(false);
                    }
                    node = join.ok_or("missing lexical switch join")?;
                }
                ControlFlow::Return => {
                    scope.register(Branch::Return);
                    return Ok(false);
                }
                ControlFlow::ReturnValue(value) => {
                    scope.register(Branch::ReturnValue(value));
                    return Ok(false);
                }
                ControlFlow::Unreachable => {
                    scope.register(Branch::Unreachable);
                    return Ok(false);
                }
                ControlFlow::Loop { .. } | ControlFlow::LoopBreak { .. } => unreachable!(),
            }
        }
    }
}
