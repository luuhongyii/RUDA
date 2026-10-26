use std::mem::take;

use petgraph::visit::EdgeRef;
use ruda_core::ir::{Instruction, Operation, PointerValueLowering, Variable};

use super::{AtomicCounter, Optimizer, PhiInstruction, SharedLiveness, version::PhiEntry};
use super::analyses::liveness::Liveness;
use super::passes::{DisaggregateArray, OptimizerPass};

impl Optimizer {
    /// Lower optimized pointer SSA at a backend boundary using the existing logical storage ABI.
    pub fn lower_pointer_values(&mut self) -> Result<(), String> {
        let nodes = self.node_ids();
        let present = nodes.iter().any(|node| {
            let block = self.block(*node);
            block
                .phi_nodes
                .borrow()
                .iter()
                .any(|phi| phi.out.ty.is_pointer())
                || block.ops.borrow().values().any(|inst| {
                    matches!(inst.operation, Operation::Pointer(_))
                        || inst.out.is_some_and(|out| out.ty.is_pointer())
                })
        });
        if !present {
            return Ok(());
        }

        let mut definitions = self.root_scope.child();
        for node in &nodes {
            let block = self.block(*node);
            definitions
                .instructions
                .extend(block.ops.borrow().values().cloned());
            for phi in block
                .phi_nodes
                .borrow()
                .iter()
                .filter(|phi| phi.out.ty.is_pointer())
            {
                definitions.instructions.extend(
                    phi.entries
                        .iter()
                        .map(|entry| Instruction::new(Operation::Copy(entry.value), phi.out)),
                );
            }
        }
        let mut lowering = PointerValueLowering::new(&definitions)?;

        for node in nodes {
            let mut scope = self.root_scope.child();
            let old_phis = take(&mut *self.program[node].phi_nodes.borrow_mut());
            let mut phis = Vec::new();
            for phi in old_phis {
                if !phi.out.ty.is_pointer() {
                    phis.push(phi);
                    continue;
                }
                let outputs = lowering.components(&scope, phi.out);
                let inputs = phi
                    .entries
                    .iter()
                    .map(|entry| (entry.block, lowering.components(&scope, entry.value)))
                    .collect::<Vec<_>>();
                for (index, out) in outputs.into_iter().enumerate() {
                    phis.push(PhiInstruction {
                        out,
                        entries: inputs
                            .iter()
                            .map(|(block, values)| PhiEntry {
                                block: *block,
                                value: values
                                    .get(index)
                                    .copied()
                                    .unwrap_or_else(|| Variable::from(0u64)),
                            })
                            .collect(),
                    });
                }
            }
            *self.program[node].phi_nodes.borrow_mut() = phis;
            if !self.program[node].ops.borrow().values().any(|inst| {
                matches!(inst.operation, Operation::Pointer(_))
                    || inst.out.is_some_and(|out| out.ty.is_pointer())
            }) {
                continue;
            }
            scope.instructions = take(&mut *self.program[node].ops.borrow_mut())
                .into_iter()
                .map(|(_, inst)| inst)
                .collect();
            lowering.lower_scope(&mut scope)?;

            let edges = self
                .program
                .edges(node)
                .map(|edge| (edge.id(), edge.target(), *edge.weight()))
                .collect::<Vec<_>>();
            for (edge, _, _) in &edges {
                self.program.remove_edge(*edge);
            }
            let control = take(&mut *self.program[node].control_flow.borrow_mut());
            let scope_join = self.program[node].scope_join.take();
            self.current_block = Some(node);
            self.parse_scope(scope);
            let end = self
                .current_block
                .expect("pointer lowering preserves fallthrough");
            *self.program[end].control_flow.borrow_mut() = control;
            self.program[end].scope_join = scope_join;
            for (_, target, weight) in edges {
                self.program.add_edge(end, target, weight);
                for phi in self.program[target].phi_nodes.borrow_mut().iter_mut() {
                    for entry in &mut phi.entries {
                        if entry.block == node {
                            entry.block = end;
                        }
                    }
                }
            }
        }
        self.current_block = None;
        self.invalidate_structure();
        self.invalidate_analysis::<SharedLiveness>();
        self.split_critical_edges();
        self.ssa_transform();
        self.apply_post_ssa_passes();
        let arrays = AtomicCounter::new(0);
        DisaggregateArray.apply_post_ssa(self, arrays.clone());
        if arrays.get() > 0 {
            self.invalidate_analysis::<Liveness>();
            self.ssa_transform();
            self.apply_post_ssa_passes();
        }
        self.split_free();
        self.invalidate_analysis::<SharedLiveness>();
        self.analysis::<SharedLiveness>();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruda_core::{ir::*, launch::RudaDim};

    fn address(scope: &mut Scope, buffer: u32, access: PointerAccess) -> Variable {
        let ty = UIntKind::U32.into();
        let pointer = TypedPointer {
            base: Variable::new(VariableKind::GlobalOutputArray(buffer), ty),
            index: 0u32.into(),
            ty: PointerType {
                pointee: ty,
                address_space: AddressSpace::Global,
                access,
            },
            checked: true,
            unroll_factor: 1,
        };
        let out = *scope.create_local(pointer.ty.value_type());
        scope.register(Instruction::new(PointerOp::Address(pointer), out));
        out
    }

    fn optimize(scope: Scope) -> Optimizer {
        OptimizerBuilder::default()
            .with_pointer_values()
            .optimize(scope, RudaDim::new_1d(4))
    }

    use super::super::OptimizerBuilder;

    #[test]
    fn pointer_ssa_retains_loop_phi_and_lowers_at_backend_boundary() {
        let mut scope = Scope::root(false);
        scope.register_type::<usize>(UIntKind::U32.into());
        let first = address(&mut scope, 0, PointerAccess::ReadOnly);
        let second = address(&mut scope, 1, PointerAccess::ReadOnly);
        let dest = address(&mut scope, 2, PointerAccess::ReadWrite);
        let pointer = *scope.create_local_restricted(first.ty);
        scope.add_local_mut(pointer);
        scope.register(Instruction::new(Operation::Copy(first), pointer));
        let mut body = scope.child();
        let next = *body.create_local(pointer.ty);
        let condition = *body.create_local(ElemType::Bool.into());
        body.register(Instruction::new(
            Comparison::Equal(BinaryOperator {
                lhs: Variable::builtin(Builtin::UnitPos, UIntKind::U32.into()),
                rhs: 0u32.into(),
            }),
            condition,
        ));
        body.register(Instruction::new(
            PointerOp::Select(Select {
                cond: condition,
                then: pointer,
                or_else: second,
            }),
            next,
        ));
        body.register(Instruction::new(Operation::Copy(next), pointer));
        let value = *body.create_local(UIntKind::U32.into());
        body.register(Instruction::new(
            PointerOp::Load {
                pointer,
                checked: true,
            },
            value,
        ));
        body.register(Instruction::no_out(PointerOp::Store {
            pointer: dest,
            value,
            checked: true,
        }));
        let advanced = *body.create_local(pointer.ty);
        body.register(Instruction::new(
            PointerOp::Offset(BinaryOperator {
                lhs: pointer,
                rhs: 1u64.into(),
            }),
            advanced,
        ));
        body.register(Instruction::new(Operation::Copy(advanced), pointer));
        let index = *scope.create_local_restricted(UIntKind::U32.into());
        scope.register(Instruction::no_out(Branch::RangeLoop(Box::new(
            RangeLoop {
                i: index,
                start: 0u32.into(),
                end: 3u32.into(),
                step: None,
                inclusive: false,
                scope: body,
            },
        ))));
        let mut opt = optimize(scope);
        for node in opt.node_ids() {
            for inst in opt.block(node).ops.borrow().values() {
                if let Operation::Arithmetic(Arithmetic::Add(op)) = &inst.operation {
                    assert_eq!(op.lhs.ty, op.rhs.ty);
                }
            }
        }
        assert!(opt.node_ids().iter().any(|node| {
            opt.block(*node)
                .phi_nodes
                .borrow()
                .iter()
                .any(|phi| phi.out.ty.is_pointer())
        }));
        opt.lower_pointer_values().unwrap();
        for node in opt.node_ids() {
            let block = opt.block(node);
            for phi in block.phi_nodes.borrow().iter() {
                assert!(!phi.out.ty.is_pointer());
                assert!(
                    phi.entries
                        .iter()
                        .all(|entry| opt.predecessors(node).contains(&entry.block))
                );
            }
            for inst in block.ops.borrow().values() {
                assert!(!matches!(inst.operation, Operation::Pointer(_)));
                assert!(!inst.out.is_some_and(|out| out.ty.is_pointer()));
            }
        }
    }

    #[test]
    fn pointer_ssa_loads_are_not_commoned_across_a_store() {
        let mut scope = Scope::root(false);
        let pointer = address(&mut scope, 0, PointerAccess::ReadWrite);
        let output = address(&mut scope, 1, PointerAccess::ReadWrite);
        let before = *scope.create_local(UIntKind::U32.into());
        let after = *scope.create_local(UIntKind::U32.into());
        let sum = *scope.create_local(UIntKind::U32.into());
        scope.register(Instruction::new(
            PointerOp::Load {
                pointer,
                checked: true,
            },
            before,
        ));
        scope.register(Instruction::no_out(PointerOp::Store {
            pointer,
            value: 7u32.into(),
            checked: true,
        }));
        scope.register(Instruction::new(
            PointerOp::Load {
                pointer,
                checked: true,
            },
            after,
        ));
        scope.register(Instruction::new(
            Arithmetic::Add(BinaryOperator {
                lhs: before,
                rhs: after,
            }),
            sum,
        ));
        scope.register(Instruction::no_out(PointerOp::Store {
            pointer: output,
            value: sum,
            checked: true,
        }));
        let mut opt = optimize(scope);
        assert_eq!(
            opt.node_ids()
                .iter()
                .map(|node| opt
                    .block(*node)
                    .ops
                    .borrow()
                    .values()
                    .filter(|inst| matches!(
                        inst.operation,
                        Operation::Pointer(PointerOp::Load { .. })
                    ))
                    .count())
                .sum::<usize>(),
            2
        );
        opt.lower_pointer_values().unwrap();
    }

    #[test]
    fn pointer_store_prevents_delaying_an_indexed_read_into_a_copy() {
        let mut scope = Scope::root(false);
        let pointer = address(&mut scope, 0, PointerAccess::ReadWrite);
        let ty = UIntKind::U32.into();
        let input = Variable::new(VariableKind::GlobalOutputArray(0), ty);
        let output = Variable::new(VariableKind::GlobalOutputArray(1), ty);
        let value = *scope.create_local(ty);
        scope.register(Instruction::new(
            Operator::Index(IndexOperator {
                list: input,
                index: 0u32.into(),
                vector_size: 0,
                unroll_factor: 1,
            }),
            value,
        ));
        scope.register(Instruction::no_out(PointerOp::Store {
            pointer,
            value: 7u32.into(),
            checked: true,
        }));
        scope.register(Instruction::new(
            Operator::IndexAssign(IndexAssignOperator {
                index: 0u32.into(),
                value,
                vector_size: 0,
                unroll_factor: 1,
            }),
            output,
        ));
        let mut opt = optimize(scope);
        assert!(opt.node_ids().iter().any(|node| {
            opt.block(*node)
                .ops
                .borrow()
                .values()
                .any(|inst| matches!(inst.operation, Operation::Operator(Operator::Index(_))))
        }));
        opt.lower_pointer_values().unwrap();
    }
}
