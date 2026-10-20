use super::{Flag, Param, ParamId, Parameter, transform::{ParamTransform, SharedTransform}};
use crate::module::{ModuleMapper, ModuleVisitor};
use alloc::boxed::Box;
use alloc::string::String;
use core::{any::Any, marker::PhantomData};
use ruda_tensor::{Backend, Bool, Int, Tensor};
use ruda_tensor::primitive::ErasedTensorPrimitive;

macro_rules! dispatch_rank {
    ($rank:expr, $d:ident => $body:block) => {
        match $rank {
            0 => { const $d: usize = 0; $body }
            1 => { const $d: usize = 1; $body }
            2 => { const $d: usize = 2; $body }
            3 => { const $d: usize = 3; $body }
            4 => { const $d: usize = 4; $body }
            5 => { const $d: usize = 5; $body }
            6 => { const $d: usize = 6; $body }
            7 => { const $d: usize = 7; $body }
            8 => { const $d: usize = 8; $body }
            other => panic!("Unsupported reparameterization tensor rank: {other}"),
        }
    };
}

pub(crate) trait DynVisitor {
    fn float(&mut self, rank: usize, param: &dyn Any);
    fn int(&mut self, rank: usize, param: &dyn Any);
    fn bool(&mut self, rank: usize, param: &dyn Any);
    fn flag(&mut self, param: &Param<Flag>);
    fn float_path(&mut self, rank: usize, path: &[String], id: ParamId, tensor: &dyn Any);
    fn int_path(&mut self, rank: usize, path: &[String], id: ParamId, tensor: &dyn Any);
    fn bool_path(&mut self, rank: usize, path: &[String], id: ParamId, tensor: &dyn Any);
    fn enter(&mut self, name: &str, kind: &str);
    fn exit(&mut self, name: &str, kind: &str);
}

pub(crate) trait DynMapper {
    fn tensor(&mut self, rank: usize, tensor: Box<dyn Any + Send>) -> Box<dyn Any + Send>;
    fn float(&mut self, rank: usize, param: Box<dyn Any + Send>) -> Box<dyn Any + Send>;
    fn int(&mut self, rank: usize, param: Box<dyn Any + Send>) -> Box<dyn Any + Send>;
    fn bool(&mut self, rank: usize, param: Box<dyn Any + Send>) -> Box<dyn Any + Send>;
    fn flag(&mut self, param: Param<Flag>) -> Param<Flag>;
    fn enter(&mut self, name: &str, kind: &str);
    fn exit(&mut self, name: &str, kind: &str);
    fn attaches(&self) -> bool;
    fn materializes(&self) -> bool;
}

pub(crate) struct VisitorToDyn<'a>(pub &'a mut dyn DynVisitor);
impl<B: Backend> ModuleVisitor<B> for VisitorToDyn<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) { self.0.float(D, param); }
    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, Int>>) { self.0.int(D, param); }
    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<B, D, Bool>>) { self.0.bool(D, param); }
    fn visit_flag(&mut self, param: &Param<Flag>) { self.0.flag(param); }
    fn visit_float_with_path<const D: usize>(&mut self, path: &[String], id: ParamId, tensor: &Tensor<B, D>) {
        self.0.float_path(D, path, id, tensor);
    }
    fn visit_int_with_path<const D: usize>(&mut self, path: &[String], id: ParamId, tensor: &Tensor<B, D, Int>) {
        self.0.int_path(D, path, id, tensor);
    }
    fn visit_bool_with_path<const D: usize>(&mut self, path: &[String], id: ParamId, tensor: &Tensor<B, D, Bool>) {
        self.0.bool_path(D, path, id, tensor);
    }
    fn enter_module(&mut self, name: &str, kind: &str) { self.0.enter(name, kind); }
    fn exit_module(&mut self, name: &str, kind: &str) { self.0.exit(name, kind); }
}

pub(crate) struct MapperToDyn<'a>(pub &'a mut dyn DynMapper);
impl<B: Backend> ModuleMapper<B> for MapperToDyn<'_> {
    fn map_tensor<const D: usize, K: ruda_tensor::BasicOps<B>>(&mut self, tensor: Tensor<B, D, K>) -> Tensor<B, D, K> {
        let tensor = K::erase_primitive(tensor.into_primitive());
        let tensor = *self.0.tensor(D, Box::new(tensor)).downcast::<ErasedTensorPrimitive<B>>()
            .expect("Reparameterization tensor backend mismatch");
        Tensor::from_primitive(K::restore_primitive(tensor))
    }
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        *self.0.float(D, Box::new(param)).downcast().expect("Reparameterization float parameter type mismatch")
    }
    fn map_int<const D: usize>(&mut self, param: Param<Tensor<B, D, Int>>) -> Param<Tensor<B, D, Int>> {
        *self.0.int(D, Box::new(param)).downcast().expect("Reparameterization integer parameter type mismatch")
    }
    fn map_bool<const D: usize>(&mut self, param: Param<Tensor<B, D, Bool>>) -> Param<Tensor<B, D, Bool>> {
        *self.0.bool(D, Box::new(param)).downcast().expect("Reparameterization boolean parameter type mismatch")
    }
    fn map_flag(&mut self, param: Param<Flag>) -> Param<Flag> { self.0.flag(param) }
    fn enter_module(&mut self, name: &str, kind: &str) { self.0.enter(name, kind); }
    fn exit_module(&mut self, name: &str, kind: &str) { self.0.exit(name, kind); }
    fn attaches_reparameterizations(&self) -> bool { self.0.attaches() }
    fn materialize_reparameterizations(&self) -> bool { self.0.materializes() }
}

struct VisitorFromDyn<'a, B, V> { visitor: &'a mut V, backend: PhantomData<B> }
impl<B: Backend, V: ModuleVisitor<B>> DynVisitor for VisitorFromDyn<'_, B, V> {
    fn float(&mut self, rank: usize, param: &dyn Any) {
        dispatch_rank!(rank, D => { self.visitor.visit_float(param.downcast_ref::<Param<Tensor<B, D>>>().unwrap()); })
    }
    fn int(&mut self, rank: usize, param: &dyn Any) {
        dispatch_rank!(rank, D => { self.visitor.visit_int(param.downcast_ref::<Param<Tensor<B, D, Int>>>().unwrap()); })
    }
    fn bool(&mut self, rank: usize, param: &dyn Any) {
        dispatch_rank!(rank, D => { self.visitor.visit_bool(param.downcast_ref::<Param<Tensor<B, D, Bool>>>().unwrap()); })
    }
    fn flag(&mut self, param: &Param<Flag>) { self.visitor.visit_flag(param); }
    fn float_path(&mut self, rank: usize, path: &[String], id: ParamId, tensor: &dyn Any) {
        dispatch_rank!(rank, D => { self.visitor.visit_float_with_path(path, id, tensor.downcast_ref::<Tensor<B, D>>().unwrap()); })
    }
    fn int_path(&mut self, rank: usize, path: &[String], id: ParamId, tensor: &dyn Any) {
        dispatch_rank!(rank, D => { self.visitor.visit_int_with_path(path, id, tensor.downcast_ref::<Tensor<B, D, Int>>().unwrap()); })
    }
    fn bool_path(&mut self, rank: usize, path: &[String], id: ParamId, tensor: &dyn Any) {
        dispatch_rank!(rank, D => { self.visitor.visit_bool_with_path(path, id, tensor.downcast_ref::<Tensor<B, D, Bool>>().unwrap()); })
    }
    fn enter(&mut self, name: &str, kind: &str) { self.visitor.enter_module(name, kind); }
    fn exit(&mut self, name: &str, kind: &str) { self.visitor.exit_module(name, kind); }
}

struct MapperFromDyn<'a, B, M> { mapper: &'a mut M, backend: PhantomData<B> }
impl<B: Backend, M: ModuleMapper<B>> DynMapper for MapperFromDyn<'_, B, M> {
    fn tensor(&mut self, rank: usize, tensor: Box<dyn Any + Send>) -> Box<dyn Any + Send> {
        let tensor = *tensor.downcast::<ErasedTensorPrimitive<B>>().expect("Reparameterization tensor backend mismatch");
        dispatch_rank!(rank, D => {
            Box::new(match tensor {
                ErasedTensorPrimitive::Float(tensor) => ErasedTensorPrimitive::<B>::Float(
                    self.mapper.map_tensor(Tensor::<B, D>::from_primitive(tensor)).into_primitive()),
                ErasedTensorPrimitive::Int(tensor) => ErasedTensorPrimitive::<B>::Int(
                    self.mapper.map_tensor(Tensor::<B, D, Int>::from_primitive(tensor)).into_primitive()),
                ErasedTensorPrimitive::Bool(tensor) => ErasedTensorPrimitive::<B>::Bool(
                    self.mapper.map_tensor(Tensor::<B, D, Bool>::from_primitive(tensor)).into_primitive()),
            })
        })
    }
    fn float(&mut self, rank: usize, param: Box<dyn Any + Send>) -> Box<dyn Any + Send> {
        dispatch_rank!(rank, D => { Box::new(self.mapper.map_float(*param.downcast::<Param<Tensor<B, D>>>().unwrap())) })
    }
    fn int(&mut self, rank: usize, param: Box<dyn Any + Send>) -> Box<dyn Any + Send> {
        dispatch_rank!(rank, D => { Box::new(self.mapper.map_int(*param.downcast::<Param<Tensor<B, D, Int>>>().unwrap())) })
    }
    fn bool(&mut self, rank: usize, param: Box<dyn Any + Send>) -> Box<dyn Any + Send> {
        dispatch_rank!(rank, D => { Box::new(self.mapper.map_bool(*param.downcast::<Param<Tensor<B, D, Bool>>>().unwrap())) })
    }
    fn flag(&mut self, param: Param<Flag>) -> Param<Flag> { self.mapper.map_flag(param) }
    fn enter(&mut self, name: &str, kind: &str) { self.mapper.enter_module(name, kind); }
    fn exit(&mut self, name: &str, kind: &str) { self.mapper.exit_module(name, kind); }
    fn attaches(&self) -> bool { self.mapper.attaches_reparameterizations() }
    fn materializes(&self) -> bool { self.mapper.materialize_reparameterizations() }
}

pub(crate) fn visit<B: Backend, T: Parameter, V: ModuleVisitor<B>>(transform: &dyn ParamTransform<T>, visitor: &mut V) {
    transform.visit_dyn(&mut VisitorFromDyn::<B, V> { visitor, backend: PhantomData });
}

pub(crate) fn map<B: Backend, T: Parameter, M: ModuleMapper<B>>(transform: &dyn ParamTransform<T>, mapper: &mut M) -> SharedTransform<T> {
    transform.map_dyn(&mut MapperFromDyn::<B, M> { mapper, backend: PhantomData })
}
