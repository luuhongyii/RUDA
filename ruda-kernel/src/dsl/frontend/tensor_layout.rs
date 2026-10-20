use alloc::vec::Vec;

use crate::dsl::{ir, prelude::*, unexpanded};
use ruda_core::ir::{CoopMma, Instruction, ManagedVariable, SemanticType};

#[derive_ruda_comptime]
pub enum TensorClampMode {
    Undefined,
    Constant(u32),
    ClampToEdge,
    Repeat,
    RepeatMirrored,
}

impl From<TensorClampMode> for ir::TensorClampMode {
    fn from(mode: TensorClampMode) -> Self {
        match mode {
            TensorClampMode::Undefined => Self::Undefined,
            TensorClampMode::Constant(value) => Self::Constant(value),
            TensorClampMode::ClampToEdge => Self::ClampToEdge,
            TensorClampMode::Repeat => Self::Repeat,
            TensorClampMode::RepeatMirrored => Self::RepeatMirrored,
        }
    }
}

/// Tensor addressing metadata. Construct through [`TensorView::new`].
#[derive(Clone, Copy)]
pub struct TensorLayout;

/// A compile-time permutation applied when loading or storing a matrix.
#[derive(Clone, Copy)]
pub struct TensorReinterpret;

#[derive(Clone)]
pub struct TensorLayoutExpand {
    pub(crate) variable: ManagedVariable,
}

#[derive(Clone)]
pub struct TensorReinterpretExpand {
    pub(crate) variable: ManagedVariable,
}

impl RudaType for TensorLayout {
    type ExpandType = TensorLayoutExpand;
}
impl RudaType for TensorReinterpret {
    type ExpandType = TensorReinterpretExpand;
}
impl IntoMut for TensorLayoutExpand {
    fn into_mut(self, _: &mut Scope) -> Self { self }
}
impl IntoMut for TensorReinterpretExpand {
    fn into_mut(self, _: &mut Scope) -> Self { self }
}
impl RudaDebug for TensorLayoutExpand {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        scope.update_variable_name(*self.variable, name);
    }
}
impl RudaDebug for TensorReinterpretExpand {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        scope.update_variable_name(*self.variable, name);
    }
}

/// A buffer and tensor addressing metadata; slicing/permutation do not copy data.
#[derive(RudaType, Clone)]
pub struct TensorView<T: RudaPrimitive> {
    pub(crate) buffer: Slice<T>,
    pub(crate) layout: TensorLayout,
    pub(crate) view: ComptimeOption<TensorReinterpret>,
}

#[derive(RudaType)]
pub struct TensorViewBuilder<T: RudaPrimitive> {
    buffer: Slice<T>,
    shape: Sequence<u32>,
    strides: ComptimeOption<Sequence<u32>>,
    #[ruda(comptime)]
    clamp_mode: TensorClampMode,
}

impl<T: RudaPrimitive> TensorView<T> {
    pub fn new(_buffer: &Slice<T>, _shape: Sequence<u32>) -> TensorViewBuilder<T> {
        unexpanded!()
    }

    pub fn __expand_new(
        _scope: &mut Scope,
        buffer: SliceExpand<T, ReadOnly>,
        shape: SequenceExpand<u32>,
    ) -> TensorViewBuilderExpand<T> {
        TensorViewBuilderExpand {
            buffer,
            shape,
            strides: ComptimeOptionExpand::None,
            clamp_mode: TensorClampMode::Constant(0),
        }
    }

    pub fn slice(&self, _offsets: Sequence<u32>, _shape: Sequence<u32>) -> Self {
        unexpanded!()
    }

    pub fn permuted(&self, _permutation: Sequence<usize>) -> Self {
        unexpanded!()
    }
}

impl<T: RudaPrimitive> TensorViewExpand<T> {
    pub fn __expand_slice_method(
        self,
        scope: &mut Scope,
        offsets: SequenceExpand<u32>,
        shape: SequenceExpand<u32>,
    ) -> Self {
        let layout_ty = match self.layout.variable.ty {
            ir::Type::Semantic(SemanticType::TensorLayout(layout)) => layout,
            _ => unreachable!(),
        };
        assert_eq!(offsets.len(), layout_ty.rank as usize, "Offsets and layout rank must match");
        assert_eq!(shape.len(), offsets.len(), "Offsets and shape must have the same rank");
        let variable = scope.create_local(self.layout.variable.ty);
        scope.register(Instruction::new(CoopMma::SliceTensorLayout {
            layout: *self.layout.variable,
            offsets: offsets.into_iter().map(|it| *it.expand).collect(),
            shape: shape.into_iter().map(|it| *it.expand).collect(),
        }, *variable));
        Self { layout: TensorLayoutExpand { variable }, ..self }
    }

    pub fn __expand_permuted_method(
        self,
        scope: &mut Scope,
        permutation: SequenceExpand<usize>,
    ) -> Self {
        let rank = permutation.len();
        assert!((1..=5).contains(&rank), "Tensor views require 1 to 5 dimensions");
        let layout_rank = match self.layout.variable.ty {
            ir::Type::Semantic(SemanticType::TensorLayout(layout)) => layout.rank as usize,
            _ => unreachable!(),
        };
        assert_eq!(rank, layout_rank, "A permutation view must have the layout's rank");
        let mut axes = [0u32; 5];
        let mut used = [false; 5];
        for (index, axis) in permutation.into_iter().enumerate() {
            let axis = axis.constant().expect("Permutation must be constant").as_usize();
            assert!(axis < rank && !used[axis], "Invalid tensor permutation");
            axes[index] = axis as u32;
            used[axis] = true;
        }
        let ty = SemanticType::TensorView(ir::TensorViewType { rank: rank as u32, permutation: axes });
        let variable = scope.create_local(ty.into());
        scope.register(Instruction::new(CoopMma::CreateTensorView, *variable));
        Self {
            view: ComptimeOptionExpand::Some(TensorReinterpretExpand { variable }),
            ..self
        }
    }
}

impl<T: RudaPrimitive> TensorViewBuilder<T> {
    pub fn with_strides(mut self, strides: Sequence<u32>) -> Self {
        self.strides = ComptimeOption::Some(strides);
        self
    }

    pub fn with_clamp_mode(mut self, clamp_mode: TensorClampMode) -> Self {
        self.clamp_mode = clamp_mode;
        self
    }

    pub fn finish(self) -> TensorView<T> { unexpanded!() }
}

impl<T: RudaPrimitive> TensorViewBuilderExpand<T> {
    pub fn __expand_with_strides_method(
        mut self, _scope: &mut Scope, strides: SequenceExpand<u32>,
    ) -> Self {
        self.strides = ComptimeOptionExpand::Some(strides);
        self
    }

    pub fn __expand_with_clamp_mode_method(
        mut self, _scope: &mut Scope, clamp_mode: TensorClampMode,
    ) -> Self {
        self.clamp_mode = clamp_mode;
        self
    }

    pub fn __expand_finish_method(self, scope: &mut Scope) -> TensorViewExpand<T> {
        let rank = self.shape.len();
        assert!((1..=5).contains(&rank), "Tensor layouts require 1 to 5 dimensions");
        let shape = self.shape.into_iter().map(|it| *it.expand).collect::<Vec<_>>();
        let strides = match self.strides {
            ComptimeOptionExpand::None => None,
            ComptimeOptionExpand::Some(strides) => {
                assert_eq!(strides.len(), rank, "Strides and shape must have the same rank");
                Some(strides.into_iter().map(|it| *it.expand).collect())
            }
        };
        let ty = SemanticType::TensorLayout(ir::TensorLayoutType {
            rank: rank as u32, clamp_mode: self.clamp_mode.into(),
        });
        let variable = scope.create_local(ty.into());
        scope.register(Instruction::new(CoopMma::CreateTensorLayout { shape, strides }, *variable));
        TensorViewExpand {
            buffer: self.buffer,
            layout: TensorLayoutExpand { variable },
            view: ComptimeOptionExpand::None,
        }
    }
}

/// Launch-time buffer and metadata for a tensor view.
pub struct TensorViewArg<R: LaunchMarker = ()> {
    pub buffer: ArrayArg<R>,
    pub shape: SequenceArg<R, u32>,
    pub strides: Option<SequenceArg<R, u32>>,
    pub clamp_mode: TensorClampMode,
}

impl<R: LaunchMarker> TensorViewArg<R> {
    pub fn new(buffer: ArrayArg<R>, shape: SequenceArg<R, u32>) -> Self {
        Self { buffer, shape, strides: None, clamp_mode: TensorClampMode::Constant(0) }
    }

    pub fn with_strides(mut self, strides: SequenceArg<R, u32>) -> Self {
        self.strides = Some(strides);
        self
    }

    pub fn with_clamp_mode(mut self, clamp_mode: TensorClampMode) -> Self {
        self.clamp_mode = clamp_mode;
        self
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct TensorViewCompilationArg {
    buffer: ArrayCompilationArg,
    shape: SequenceCompilationArg<u32>,
    strides: Option<SequenceCompilationArg<u32>>,
    clamp_mode: TensorClampMode,
}

impl<T: RudaPrimitive> LaunchArg for TensorViewBuilder<T> {
    type RuntimeArg<R: LaunchMarker> = TensorViewArg<R>;
    type CompilationArg = TensorViewCompilationArg;

    fn register<R: LaunchMarker>(arg: Self::RuntimeArg<R>, launcher: &mut KernelLauncher<R>) -> Self::CompilationArg {
        TensorViewCompilationArg {
            buffer: Array::<T>::register(arg.buffer, launcher),
            shape: Sequence::<u32>::register(arg.shape, launcher),
            strides: arg.strides.map(|it| Sequence::<u32>::register(it, launcher)),
            clamp_mode: arg.clamp_mode,
        }
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> TensorViewBuilderExpand<T> {
        let buffer = Array::<T>::expand(&arg.buffer, builder);
        expand_builder(arg, builder, buffer)
    }

    fn expand_output(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> TensorViewBuilderExpand<T> {
        let buffer = Array::<T>::expand_output(&arg.buffer, builder);
        expand_builder(arg, builder, buffer)
    }
}

fn expand_builder<T: RudaPrimitive>(
    arg: &TensorViewCompilationArg,
    builder: &mut KernelBuilder,
    buffer: NativeExpand<Array<T>>,
) -> TensorViewBuilderExpand<T> {
    let buffer = buffer.__expand_to_slice_method(&mut builder.scope);
    let shape = Sequence::<u32>::expand(&arg.shape, builder);
    let strides = match &arg.strides {
        None => ComptimeOptionExpand::None,
        Some(strides) => ComptimeOptionExpand::Some(Sequence::<u32>::expand(strides, builder)),
    };
    TensorViewBuilderExpand { buffer, shape, strides, clamp_mode: arg.clamp_mode }
}

impl<T: RudaPrimitive> LaunchArg for TensorView<T> {
    type RuntimeArg<R: LaunchMarker> = TensorViewArg<R>;
    type CompilationArg = TensorViewCompilationArg;

    fn register<R: LaunchMarker>(arg: Self::RuntimeArg<R>, launcher: &mut KernelLauncher<R>) -> Self::CompilationArg {
        TensorViewBuilder::<T>::register(arg, launcher)
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> TensorViewExpand<T> {
        TensorViewBuilder::<T>::expand(arg, builder).__expand_finish_method(&mut builder.scope)
    }

    fn expand_output(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> TensorViewExpand<T> {
        TensorViewBuilder::<T>::expand_output(arg, builder).__expand_finish_method(&mut builder.scope)
    }
}
