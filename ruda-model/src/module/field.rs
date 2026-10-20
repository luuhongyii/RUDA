use super::{AutodiffModule, Module, ModuleMapper, ModuleVisitor};
use crate::tensor::backend::{AutodiffBackend, Backend};
use core::{cell::Cell, marker::PhantomData};

pub struct Field<T, B>(Cell<Option<T>>, PhantomData<B>);

impl<T, B> Field<T, B> {
    pub fn new(value: T) -> Self {
        Self(Cell::new(Some(value)), PhantomData)
    }

    fn take(&self) -> T {
        self.0.take().expect("module field consumed once")
    }
}

pub struct FieldRef<'a, T, B>(pub &'a T, pub PhantomData<B>);

// Autoref dispatch keeps opaque configuration fields unchanged while allowing
// backend-independent modules to participate in their parent's lifecycle.
pub trait MapField<B: Backend> {
    type Value;
    fn map_field<M: ModuleMapper<B>>(self, mapper: &mut M, name: &str, kind: &str) -> Self::Value;
}

impl<T: Module<B>, B: Backend> MapField<B> for &Field<T, B> {
    type Value = T;
    fn map_field<M: ModuleMapper<B>>(self, mapper: &mut M, name: &str, kind: &str) -> T {
        mapper.enter_module(name, kind);
        let value = self.take().map(mapper);
        mapper.exit_module(name, kind);
        value
    }
}

impl<T, B: Backend> MapField<B> for &&Field<T, B> {
    type Value = T;
    fn map_field<M: ModuleMapper<B>>(self, _: &mut M, _: &str, _: &str) -> T {
        self.take()
    }
}

pub trait VisitField<B: Backend> {
    fn visit_field<V: ModuleVisitor<B>>(self, visitor: &mut V, name: &str, kind: &str);
}

impl<T: Module<B>, B: Backend> VisitField<B> for &FieldRef<'_, T, B> {
    fn visit_field<V: ModuleVisitor<B>>(self, visitor: &mut V, name: &str, kind: &str) {
        visitor.enter_module(name, kind);
        self.0.visit(visitor);
        visitor.exit_module(name, kind);
    }
}

impl<T, B: Backend> VisitField<B> for &&FieldRef<'_, T, B> {
    fn visit_field<V: ModuleVisitor<B>>(self, _: &mut V, _: &str, _: &str) {}
}

pub trait ValidField {
    type Value;
    fn valid_field(self) -> Self::Value;
}

impl<T: AutodiffModule<B, InnerModule = T>, B: AutodiffBackend> ValidField for &FieldRef<'_, T, B> {
    type Value = T;
    fn valid_field(self) -> T {
        self.0.valid()
    }
}

impl<T: Clone, B> ValidField for &&FieldRef<'_, T, B> {
    type Value = T;
    fn valid_field(self) -> T {
        self.0.clone()
    }
}

pub trait TrainField {
    type Value;
    fn train_field(self) -> Self::Value;
}

impl<T: AutodiffModule<B, InnerModule = T>, B: AutodiffBackend> TrainField for &Field<T, B> {
    type Value = T;
    fn train_field(self) -> T {
        T::from_inner(self.take())
    }
}

impl<T, B> TrainField for &&Field<T, B> {
    type Value = T;
    fn train_field(self) -> T {
        self.take()
    }
}
