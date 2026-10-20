use alloc::{boxed::Box, collections::{BTreeMap, BTreeSet, LinkedList, VecDeque}, string::String, vec, vec::Vec};
use core::ops::{Bound, Range, RangeFrom, RangeFull, RangeTo, RangeToInclusive};
use ruda_tensor::Backend;
use serde::{Serialize, de::DeserializeOwned};
use super::{RuntimePrecision, RuntimeRecord, RuntimeRecordValue as Value};
use crate::module::ParamId;

impl<B: Backend, T: RuntimeRecord<B>> RuntimeRecord<B> for Box<T> {
    fn into_runtime(self, precision: RuntimePrecision) -> Value { (*self).into_runtime(precision) }
    fn from_runtime(value: Value, device: &B::Device) -> Self { Box::new(T::from_runtime(value, device)) }
}

macro_rules! sequence {
    ($container:ident) => {
        impl<B: Backend, T: RuntimeRecord<B>> RuntimeRecord<B> for $container<T> {
            fn into_runtime(self, precision: RuntimePrecision) -> Value {
                Value::Fields(0, self.into_iter().map(|value| value.into_runtime(precision)).collect())
            }
            fn from_runtime(value: Value, device: &B::Device) -> Self {
                let (tag, values) = value.into_variant();
                assert_eq!(tag, 0, "Sequence record tag mismatch");
                values.into_iter().map(|value| T::from_runtime(value, device)).collect()
            }
        }
    };
}
sequence!(Vec);
sequence!(VecDeque);
sequence!(LinkedList);

impl<B: Backend, T: RuntimeRecord<B>, const N: usize> RuntimeRecord<B> for [T; N] {
    fn into_runtime(self, precision: RuntimePrecision) -> Value {
        Value::Fields(0, self.into_iter().map(|value| value.into_runtime(precision)).collect())
    }
    fn from_runtime(value: Value, device: &B::Device) -> Self {
        let mut values = value.into_fields(0, N);
        core::array::from_fn(|_| T::from_runtime(values.next().unwrap(), device))
    }
}

impl<B: Backend, T: RuntimeRecord<B>> RuntimeRecord<B> for Option<T> {
    fn into_runtime(self, precision: RuntimePrecision) -> Value {
        match self { Some(value) => Value::Fields(1, vec![value.into_runtime(precision)]), None => Value::Empty }
    }
    fn from_runtime(value: Value, device: &B::Device) -> Self {
        match value {
            Value::Empty => None,
            value => Some(T::from_runtime(value.into_fields(1, 1).next().unwrap(), device)),
        }
    }
}

impl<B: Backend, T: RuntimeRecord<B>, E: RuntimeRecord<B>> RuntimeRecord<B> for Result<T, E> {
    fn into_runtime(self, precision: RuntimePrecision) -> Value {
        match self {
            Ok(value) => Value::Fields(0, vec![value.into_runtime(precision)]),
            Err(value) => Value::Fields(1, vec![value.into_runtime(precision)]),
        }
    }
    fn from_runtime(value: Value, device: &B::Device) -> Self {
        let (tag, values) = value.into_variant();
        assert_eq!(values.len(), 1, "Result record field count mismatch");
        let value = values.into_iter().next().unwrap();
        match tag { 0 => Ok(T::from_runtime(value, device)), 1 => Err(E::from_runtime(value, device)),
            _ => panic!("Invalid result record variant") }
    }
}

macro_rules! tuple {
    ($($ty:ident: $index:tt),+) => {
        impl<B: Backend, $($ty: RuntimeRecord<B>),+> RuntimeRecord<B> for ($($ty,)+) {
            fn into_runtime(self, precision: RuntimePrecision) -> Value {
                Value::Fields(0, vec![$(self.$index.into_runtime(precision)),+])
            }
            fn from_runtime(value: Value, device: &B::Device) -> Self {
                let count = [$(stringify!($ty)),+].len();
                let mut values = value.into_fields(0, count);
                ($($ty::from_runtime(values.next().unwrap(), device),)+)
            }
        }
    };
}
tuple!(T0: 0);
tuple!(T0: 0, T1: 1);
tuple!(T0: 0, T1: 1, T2: 2);
tuple!(T0: 0, T1: 1, T2: 2, T3: 3);
tuple!(T0: 0, T1: 1, T2: 2, T3: 3, T4: 4);
tuple!(T0: 0, T1: 1, T2: 2, T3: 3, T4: 4, T5: 5);
tuple!(T0: 0, T1: 1, T2: 2, T3: 3, T4: 4, T5: 5, T6: 6);
tuple!(T0: 0, T1: 1, T2: 2, T3: 3, T4: 4, T5: 5, T6: 6, T7: 7);
tuple!(T0: 0, T1: 1, T2: 2, T3: 3, T4: 4, T5: 5, T6: 6, T7: 7, T8: 8);
tuple!(T0: 0, T1: 1, T2: 2, T3: 3, T4: 4, T5: 5, T6: 6, T7: 7, T8: 8, T9: 9);

impl<B: Backend, K, T: RuntimeRecord<B>> RuntimeRecord<B> for BTreeMap<K, T>
where K: Ord + Clone + Send + Serialize + DeserializeOwned {
    fn into_runtime(self, precision: RuntimePrecision) -> Value {
        Value::Fields(0, self.into_iter().map(|(key, value)|
            Value::Fields(0, vec![Value::scalar(key), value.into_runtime(precision)])).collect())
    }
    fn from_runtime(value: Value, device: &B::Device) -> Self {
        let (tag, values) = value.into_variant();
        assert_eq!(tag, 0, "Map record tag mismatch");
        values.into_iter().map(|value| {
            let mut pair = value.into_fields(0, 2);
            (pair.next().unwrap().into_scalar(), T::from_runtime(pair.next().unwrap(), device))
        }).collect()
    }
}

impl<B: Backend, K> RuntimeRecord<B> for BTreeSet<K>
where K: Ord + Clone + Send + Serialize + DeserializeOwned {
    fn into_runtime(self, _: RuntimePrecision) -> Value { Value::scalar(self) }
    fn from_runtime(value: Value, _: &B::Device) -> Self { value.into_scalar() }
}

macro_rules! string_map {
    ($map:path) => {
        impl<B: Backend, T: RuntimeRecord<B>> RuntimeRecord<B> for $map {
            fn into_runtime(self, precision: RuntimePrecision) -> Value {
                <BTreeMap<String, T> as RuntimeRecord<B>>::into_runtime(self.into_iter().collect(), precision)
            }
            fn from_runtime(value: Value, device: &B::Device) -> Self {
                BTreeMap::<String, T>::from_runtime(value, device).into_iter().collect()
            }
        }
    };
}
string_map!(hashbrown::HashMap<String, T>);
#[cfg(feature = "std")]
string_map!(std::collections::HashMap<String, T>);

impl<B: Backend, T: RuntimeRecord<B>> RuntimeRecord<B> for hashbrown::HashMap<ParamId, T> {
    fn into_runtime(self, precision: RuntimePrecision) -> Value {
        let values: BTreeMap<String, T> = self.into_iter().map(|(id, value)| (id.serialize(), value)).collect();
        values.into_runtime(precision)
    }
    fn from_runtime(value: Value, device: &B::Device) -> Self {
        BTreeMap::<String, T>::from_runtime(value, device).into_iter()
            .map(|(id, value)| (ParamId::deserialize(&id), value)).collect()
    }
}

impl<B: Backend, T: RuntimeRecord<B>> RuntimeRecord<B> for Range<T> {
    fn into_runtime(self, precision: RuntimePrecision) -> Value { (self.start, self.end).into_runtime(precision) }
    fn from_runtime(value: Value, device: &B::Device) -> Self {
        let (start, end) = <(T, T)>::from_runtime(value, device);
        start..end
    }
}

macro_rules! endpoint {
    ($ty:ident, $field:ident) => {
        impl<B: Backend, T: RuntimeRecord<B>> RuntimeRecord<B> for $ty<T> {
            fn into_runtime(self, precision: RuntimePrecision) -> Value { self.$field.into_runtime(precision) }
            fn from_runtime(value: Value, device: &B::Device) -> Self { Self { $field: T::from_runtime(value, device) } }
        }
    };
}
endpoint!(RangeFrom, start);
endpoint!(RangeTo, end);
endpoint!(RangeToInclusive, end);

impl<B: Backend> RuntimeRecord<B> for RangeFull {
    fn into_runtime(self, _: RuntimePrecision) -> Value { Value::Empty }
    fn from_runtime(value: Value, _: &B::Device) -> Self {
        assert!(matches!(value, Value::Empty), "Expected an unbounded range record");
        ..
    }
}

impl<B: Backend, T: RuntimeRecord<B>> RuntimeRecord<B> for Bound<T> {
    fn into_runtime(self, precision: RuntimePrecision) -> Value {
        match self {
            Self::Included(value) => Value::Fields(0, vec![value.into_runtime(precision)]),
            Self::Excluded(value) => Value::Fields(1, vec![value.into_runtime(precision)]),
            Self::Unbounded => Value::Fields(2, vec![]),
        }
    }
    fn from_runtime(value: Value, device: &B::Device) -> Self {
        let (tag, values) = value.into_variant();
        match tag {
            0 | 1 => {
                assert_eq!(values.len(), 1, "Bound record field count mismatch");
                let value = T::from_runtime(values.into_iter().next().unwrap(), device);
                if tag == 0 { Self::Included(value) } else { Self::Excluded(value) }
            }
            2 => { assert!(values.is_empty(), "Unbounded record contains fields"); Self::Unbounded }
            _ => panic!("Invalid bound record variant"),
        }
    }
}
