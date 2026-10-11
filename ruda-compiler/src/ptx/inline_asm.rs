use super::{Result, emit::Emitter, invalid};
use ruda_core::ir::Variable;

/// `NonSemantic::Print` format strings starting with this marker are inline PTX instead of printf:
/// `MARKER template MARKER module`. `$0` in the template is the destination (`args[0]`), `$1..`
/// the inputs; `module` is emitted once per kernel at module scope (e.g. `.func` definitions).
pub const INLINE_ASM_MARKER: char = '\u{1}';

impl Emitter {
    pub(super) fn inline_asm(&mut self, format: &str, args: &[Variable]) -> Result<()> {
        let mut parts = format.trim_start_matches(INLINE_ASM_MARKER).splitn(2, INLINE_ASM_MARKER);
        let template = parts.next().unwrap_or_default();
        let module = parts.next().unwrap_or_default();
        let (out, inputs) = args.split_first().ok_or_else(|| invalid("inline asm requires a destination"))?;
        let mut operands = vec![self.destination(*out)?];
        for input in inputs {
            operands.push(self.value(*input)?);
        }
        if !module.is_empty() && !self.module_declarations.contains(module) {
            self.module_declarations += module;
            self.module_declarations += "\n";
        }
        let mut text = template.to_string();
        for (index, operand) in operands.iter().enumerate().rev() {
            text = text.replace(&format!("${index}"), operand);
        }
        for line in text.lines() {
            self.line(line.to_string());
        }
        Ok(())
    }
}
