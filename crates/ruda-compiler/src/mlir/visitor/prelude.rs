pub use super::Visitor;
pub use ruda_core::ir::Variable;
pub use crate::optimizer::Optimizer;
pub use melior::{
    Context, Error,
    ir::{BlockLike, RegionLike, Type, TypeLike, Value, ValueLike, operation::OperationLike},
};
pub use super::block_ops::BlockOps;

pub trait IntoType {
    fn to_type<'a>(self, context: &'a Context) -> Type<'a>;
    fn is_vectorized(&self) -> bool
    where
        Self: Sized,
    {
        false
    }
}
