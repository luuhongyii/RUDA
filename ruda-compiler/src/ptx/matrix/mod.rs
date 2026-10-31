use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::{CoopMma, Matrix, MatrixIdent, MatrixLayout, Type, Variable, VariableKind};

mod fragment;
mod memory;
mod execute;
mod registers;
mod manual;
mod indexing;
mod volta;
mod scaled;
mod blackwell;

fn descriptor(variable: Variable) -> Result<Matrix> {
    let VariableKind::Matrix { mat, .. } = variable.kind else {
        return Err(invalid("expected a matrix fragment"));
    };
    if mat.scope != ruda_core::ir::MatrixScope::Plane {
        return Err(unsupported("workgroup cooperative matrices are not supported by PTX"));
    }
    if variable.ty != Type::new(mat.storage) { return Err(invalid("matrix fragment storage type mismatch")); }
    Ok(mat)
}

fn shape(matrix: Matrix) -> String {
    format!("m{}n{}k{}", matrix.m, matrix.n, matrix.k)
}

fn layout(matrix: MatrixLayout) -> Result<&'static str> {
    match matrix {
        MatrixLayout::RowMajor => Ok("row"),
        MatrixLayout::ColMajor => Ok("col"),
        MatrixLayout::Undefined => Err(invalid("matrix operation requires an explicit layout")),
    }
}

fn same_shape(a: Matrix, b: Matrix) -> bool {
    (a.m, a.n, a.k) == (b.m, b.n, b.k)
}

impl Emitter {
    pub fn matrix(&mut self, operation: CoopMma, out: Variable) -> Result<()> {
        match &operation {
            CoopMma::ExecuteManual { matrix, .. }
            | CoopMma::ExecuteScaled { matrix, .. }
            | CoopMma::RowIndex { matrix, .. }
            | CoopMma::ColIndex { matrix, .. }
                if matrix.scope != ruda_core::ir::MatrixScope::Plane =>
            {
                return Err(unsupported("manual MMA requires subgroup scope"));
            }
            _ => {}
        }
        match operation {
            CoopMma::Blackwell { kind, args } => self.tcgen05(out, kind, args),
            CoopMma::Elementwise { .. } => {
                Err(unsupported("cooperative matrix callbacks are not supported by PTX"))
            }
            CoopMma::CreateTensorLayout { .. }
            | CoopMma::CreateTensorView
            | CoopMma::SliceTensorLayout { .. }
            | CoopMma::LoadTensor { .. }
            | CoopMma::StoreTensor { .. } => {
                Err(unsupported("cooperative matrix tensor addressing is not supported by PTX"))
            }
            CoopMma::Fill { value } => self.matrix_fill(out, value),
            CoopMma::Load { value, stride, offset, layout } => self.matrix_load(out, value, offset, stride, layout),
            CoopMma::Store { mat, stride, offset, layout } => self.matrix_store(out, mat, offset, stride, layout),
            CoopMma::Execute { mat_a, mat_b, mat_c } => self.matrix_execute(out, mat_a, mat_b, mat_c),
            CoopMma::Cast { input } => self.matrix_cast(out, input),
            CoopMma::LoadMatrix { buffer, offset, vector_size, factor, transpose } => self.matrix_transfer(out, buffer, offset, vector_size, factor, transpose, false),
            CoopMma::StoreMatrix { offset, vector_size, registers, factor, transpose } => self.matrix_transfer(registers, out, offset, vector_size, factor, transpose, true),
            CoopMma::ExecuteManual { matrix, registers_a, registers_b, registers_c } => self.matrix_manual(out, matrix, registers_a, registers_b, registers_c),
            CoopMma::ExecuteScaled { matrix, registers_a, registers_b, registers_c, scales_a, scales_b, scales_factor } => self.matrix_scaled(out, matrix, registers_a, registers_b, registers_c, scales_a, scales_b, scales_factor),
            CoopMma::RowIndex { lane_id, i, matrix } => self.matrix_index(out, lane_id, i, matrix, true),
            CoopMma::ColIndex { lane_id, i, matrix } => self.matrix_index(out, lane_id, i, matrix, false),
        }
    }
}
