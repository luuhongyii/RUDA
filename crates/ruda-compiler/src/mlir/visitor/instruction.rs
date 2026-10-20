use ruda_core::ir::Instruction;

use super::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_instruction(&mut self, instruction: &Instruction) {
        let instruction = instruction.clone().lower_typed_memory().expect("valid typed memory instruction");
        match instruction.out {
            Some(out) => {
                self.visit_operation_with_out(&instruction.operation, out);
            }
            None => {
                self.visit_operation(&instruction.operation);
            }
        }
    }
}
