use super::{Result, emit::Emitter, types::Scalar, unsupported};
use ruda_core::ir::{ConstantValue, CoopMma, Instruction, Operation, OperationReflect, Operator};

pub(super) fn constant(value: ConstantValue) -> String {
    let value = value.as_f64() as f32;
    ruda_core::e2m1x2::from_f32_slice(&[value, value])[0].to_bits().to_string()
}

impl Emitter {
    pub(super) fn packed_instruction_supported(&self, instruction: &Instruction) -> Result<()> {
        let args = instruction.operation.args().unwrap_or_default();
        if !instruction.out.into_iter().chain(args.iter().copied())
            .any(|v| Scalar::is_packed_fp4(v.ty))
        {
            return Ok(());
        }
        if matches!(&instruction.operation,
            Operation::Copy(_) | Operation::Metadata(_)
            | Operation::CoopMma(CoopMma::ExecuteScaled { .. })
            | Operation::Operator(Operator::Reinterpret(_) | Operator::Select(_)
                | Operator::InitVector(_) | Operator::Index(_) | Operator::UncheckedIndex(_)
                | Operator::IndexAssign(_) | Operator::UncheckedIndexAssign(_)
                | Operator::CopyMemory(_) | Operator::CopyMemoryBulk(_))
        ) {
            Ok(())
        } else {
            Err(unsupported("packed E2M1x2 supports bit-preserving transport and scaled MMA, not scalar arithmetic/conversion"))
        }
    }
}
