use alloc::{boxed::Box, format, string::String, vec, vec::Vec};

use super::Variable;
use crate::ir::{OperationCode, OperationReflect};
use crate::ir::{StorageType, TypeHash};
use core::fmt::Display;

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum MatrixIdent {
    A,
    B,
    Accumulator,
}

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum MatrixLayout {
    ColMajor,
    RowMajor,
    Undefined,
}

/// Invocations that collectively own and operate on a matrix.
#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Default, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MatrixScope {
    /// One subgroup.
    #[default]
    Plane,
    /// All invocations in the workgroup.
    Ruda,
}

/// Addressing behavior outside the dimensions of a tensor layout.
#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TensorClampMode {
    Undefined,
    Constant(u32),
    ClampToEdge,
    Repeat,
    RepeatMirrored,
}

/// Compile-time tensor layout type. Dimensions and strides are runtime values.
#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorLayoutType {
    pub rank: u32,
    pub clamp_mode: TensorClampMode,
}

/// A permutation view over a tensor layout, without independent dimensions.
#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorViewType {
    pub rank: u32,
    /// Only the first `rank` entries are used.
    pub permutation: [u32; 5],
}

#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(missing_docs)]
pub struct Matrix {
    pub ident: MatrixIdent,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub storage: StorageType,
    pub layout: MatrixLayout,
    #[cfg_attr(feature = "ir-serde", serde(default))]
    pub scope: MatrixScope,
}

impl Matrix {
    /// Create a subgroup matrix descriptor.
    pub fn new(
        ident: MatrixIdent,
        m: usize,
        n: usize,
        k: usize,
        storage: StorageType,
        layout: MatrixLayout,
    ) -> Self {
        Self { ident, m, n, k, storage, layout, scope: MatrixScope::Plane }
    }

    /// Set the collective scope without changing the matrix shape or element type.
    pub fn with_scope(mut self, scope: MatrixScope) -> Self {
        self.scope = scope;
        self
    }

    /// Number of elements in terms of the physical storage type, accounting for packed elements
    pub fn num_elems(&self) -> usize {
        let elems = match self.ident {
            MatrixIdent::A => self.m * self.k,
            MatrixIdent::B => self.k * self.n,
            MatrixIdent::Accumulator => self.m * self.n,
        };
        elems / self.storage.packing_factor()
    }
}

/// Cooperative Matrix-Multiply and Accumulate Instruction.
#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationCode)]
#[operation(opcode_name = CmmaOpCode)]
#[allow(missing_docs)]
pub enum CoopMma {
    /// Fill the matrix with the value.
    Fill { value: Variable },
    /// Load the value into the matrix given the stride.
    Load {
        value: Variable,
        stride: Variable,
        offset: Variable,
        layout: Option<MatrixLayout>,
    },
    /// Executes D=A*B+C;
    ///
    /// For implementing a matmul, `D=C` : `C+=A*B`
    Execute {
        mat_a: Variable,
        mat_b: Variable,
        mat_c: Variable,
    },
    /// Store the matrix in an output variable following the stride and the layout.
    Store {
        mat: Variable,
        stride: Variable,
        offset: Variable,
        layout: MatrixLayout,
    },
    /// Cast a fragment to another type.
    Cast { input: Variable },
    /// Invoke a scalar closure for every (row, column, component) of a fragment.
    Elementwise {
        input: Variable,
        function: Box<super::ScalarFunction>,
        captures: Vec<Variable>,
    },

    /// Create a tensor layout with contiguous or explicit element strides.
    CreateTensorLayout {
        shape: Vec<Variable>,
        strides: Option<Vec<Variable>>,
    },
    /// Create the permutation view specified by the output type.
    CreateTensorView,
    /// Add offsets to a layout and replace its spans, leaving its strides intact.
    SliceTensorLayout {
        layout: Variable,
        offsets: Vec<Variable>,
        shape: Vec<Variable>,
    },
    /// Load a matrix using tensor addressing.
    LoadTensor {
        buffer: Variable,
        offset: Variable,
        layout: Variable,
        view: Option<Variable>,
    },
    /// Store a matrix using tensor addressing into the output buffer.
    StoreTensor {
        mat: Variable,
        offset: Variable,
        layout: Variable,
        view: Option<Variable>,
    },

    /// Row index of nth element in the lane
    RowIndex {
        lane_id: Variable,
        i: Variable,
        matrix: Matrix,
    },
    /// Column index of nth element in the lane
    ColIndex {
        lane_id: Variable,
        i: Variable,
        matrix: Matrix,
    },
    /// Execute a CUDA `ldmatrix` instruction
    LoadMatrix {
        buffer: Variable,
        offset: Variable,
        vector_size: Option<usize>,
        factor: usize,
        transpose: bool,
    },
    /// Execute a CUDA `stmatrix` instruction
    StoreMatrix {
        offset: Variable,
        vector_size: Option<usize>,
        registers: Variable,
        factor: usize,
        transpose: bool,
    },
    /// Manual execute.
    ExecuteManual {
        matrix: Matrix,
        registers_a: Variable,
        registers_b: Variable,
        registers_c: Variable,
    },
    /// Scaled manual execute.
    ExecuteScaled {
        matrix: Matrix,
        registers_a: Variable,
        registers_b: Variable,
        registers_c: Variable,
        scales_a: Variable,
        scales_b: Variable,
        scales_factor: usize,
    },
}

impl OperationReflect for CoopMma {
    type OpCode = CmmaOpCode;

    fn op_code(&self) -> Self::OpCode {
        self.__match_opcode()
    }

    fn args(&self) -> Option<Vec<Variable>> {
        match self {
            CoopMma::Fill { value } => Some(vec![*value]),
            CoopMma::Load { .. }
            | CoopMma::Execute { .. }
            | CoopMma::ExecuteManual { .. }
            | CoopMma::ExecuteScaled { .. }
            | CoopMma::Store { .. }
            | CoopMma::RowIndex { .. }
            | CoopMma::ColIndex { .. }
            | CoopMma::LoadMatrix { .. }
            | CoopMma::StoreMatrix { .. }
            | CoopMma::CreateTensorLayout { .. }
            | CoopMma::CreateTensorView
            | CoopMma::SliceTensorLayout { .. }
            | CoopMma::LoadTensor { .. }
            | CoopMma::StoreTensor { .. } => None,
            CoopMma::Elementwise { .. } => None,
            CoopMma::Cast { input } => Some(vec![*input]),
        }
    }

    fn from_code_and_args(op_code: Self::OpCode, args: &[Variable]) -> Option<Self> {
        match op_code {
            CmmaOpCode::Fill => Some(CoopMma::Fill { value: args[0] }),
            CmmaOpCode::Load
            | CmmaOpCode::Execute
            | CmmaOpCode::ExecuteManual
            | CmmaOpCode::ExecuteScaled
            | CmmaOpCode::Store
            | CmmaOpCode::RowIndex
            | CmmaOpCode::ColIndex
            | CmmaOpCode::LoadMatrix
            | CmmaOpCode::StoreMatrix
            | CmmaOpCode::CreateTensorLayout
            | CmmaOpCode::CreateTensorView
            | CmmaOpCode::SliceTensorLayout
            | CmmaOpCode::LoadTensor
            | CmmaOpCode::StoreTensor => None,
            CmmaOpCode::Elementwise => None,
            CmmaOpCode::Cast => Some(CoopMma::Cast { input: args[0] }),
        }
    }
}

impl Display for CoopMma {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CoopMma::Fill { value } => write!(f, "{value}"),
            CoopMma::Load {
                value,
                stride,
                offset,
                layout,
            } => {
                let layout = layout
                    .map(|it| format!(", layout: {it:?}"))
                    .unwrap_or(String::new());
                write!(
                    f,
                    "matrix_load({value}, stride: {stride}{layout}, offset: {offset})"
                )
            }
            CoopMma::Execute {
                mat_a,
                mat_b,
                mat_c,
            } => write!(f, "execute_cmma({mat_a}, {mat_b}, {mat_c})"),
            CoopMma::ExecuteManual {
                matrix,
                registers_a,
                registers_b,
                registers_c,
            } => {
                write!(
                    f,
                    "execute_manual_mma(
                    matrix: {matrix:?},
                    frag_a: {registers_a},
                    frag_b: {registers_b},
                    frag_c: {registers_c},
                )"
                )
            }
            CoopMma::ExecuteScaled {
                matrix,
                registers_a,
                registers_b,
                registers_c,
                scales_a,
                scales_b,
                scales_factor,
            } => {
                write!(
                    f,
                    "execute_scaled_mma_{scales_factor}x(
                    matrix: {matrix:?},
                    frag_a: {registers_a},
                    frag_b: {registers_b},
                    frag_c: {registers_c},
                    scales_a: {scales_a},
                    scales_b: {scales_b}
                )"
                )
            }
            CoopMma::Store {
                mat,
                stride,
                offset,
                layout,
            } => write!(
                f,
                "matrix_store({mat}, stride: {stride}, layout: {layout:?}, offset: {offset:?})"
            ),
            CoopMma::Cast { input } => {
                write!(f, "matrix_cast(input: {input})")
            }
            CoopMma::Elementwise { input, captures, .. } => {
                write!(f, "matrix_elementwise({input}, captures: {captures:?})")
            }
            CoopMma::CreateTensorLayout { shape, strides } => {
                write!(f, "tensor_layout(shape: {shape:?}, strides: {strides:?})")
            }
            CoopMma::CreateTensorView => f.write_str("tensor_view()"),
            CoopMma::SliceTensorLayout { layout, offsets, shape } => {
                write!(f, "tensor_slice({layout}, offsets: {offsets:?}, shape: {shape:?})")
            }
            CoopMma::LoadTensor { buffer, offset, layout, view } => {
                write!(f, "matrix_load_tensor({buffer}, offset: {offset}, layout: {layout}, view: {view:?})")
            }
            CoopMma::StoreTensor { mat, offset, layout, view } => {
                write!(f, "matrix_store_tensor({mat}, offset: {offset}, layout: {layout}, view: {view:?})")
            }
            CoopMma::RowIndex { lane_id, i, matrix } => {
                write!(f, "row_idx(lane_id: {lane_id}, i: {i}, matrix: {matrix:?})",)
            }
            CoopMma::ColIndex { lane_id, i, matrix } => {
                write!(f, "col_idx(lane_id: {lane_id}, i: {i}, matrix: {matrix:?})",)
            }
            CoopMma::LoadMatrix {
                buffer,
                offset,
                factor,
                transpose,
                ..
            } => {
                write!(
                    f,
                    "ldmatrix_{factor}x(&{buffer}[{offset}], transpose: {transpose})"
                )
            }
            CoopMma::StoreMatrix {
                offset,
                registers,
                factor,
                transpose,
                ..
            } => {
                write!(
                    f,
                    "stmatrix_{factor}x({registers}, offset: {offset}, transpose: {transpose})"
                )
            }
        }
    }
}
