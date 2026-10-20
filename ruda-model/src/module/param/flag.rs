use crate::module::{
    AutodiffModule, Content, Devices, EmptyRecord, HasAutodiffModule, Module, ModuleDisplay,
    ModuleDisplayDefault, ModuleMapper, ModuleVisitor, Param, ParamId, Parameter,
};
use crate::tensor::backend::{AutodiffBackend, Backend};

/// Device-independent runtime control state. Flags are omitted from model records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flag(bool);

impl Flag {
    /// Create a flag.
    pub const fn new(value: bool) -> Self {
        Self(value)
    }
    /// Whether the flag is enabled.
    pub const fn is_enabled(&self) -> bool {
        self.0
    }
    /// Return the flag with the given value.
    pub const fn with(self, value: bool) -> Self {
        Self(value)
    }
}

impl core::fmt::Display for Flag {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(if self.0 { "enabled" } else { "disabled" })
    }
}

impl Parameter for Flag {
    type Device = ();
    fn device(&self) {}
    fn is_require_grad(&self) -> bool {
        self.0
    }
    fn set_require_grad(self, enabled: bool) -> Self {
        self.with(enabled)
    }
    fn mapped_require_grad(&self, previous: bool) -> bool {
        previous
    }
}

impl Param<Flag> {
    /// Create an identified flag.
    pub fn from_bool(value: bool) -> Self {
        Self::initialized(ParamId::new(), Flag::new(value))
    }

    /// Set both the current flag and the state restored by `train`.
    pub fn with_value(self, value: bool) -> Self {
        let mut flag = self.map(|flag| flag.with(value));
        flag.require_grad = value;
        flag
    }
}

impl<B: Backend> Module<B> for Param<Flag> {
    type Record = EmptyRecord;
    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        mapper.map_flag(self)
    }
    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit_flag(self);
    }
    fn to_device(self, _: &B::Device) -> Self {
        self
    }
    fn fork(self, _: &B::Device) -> Self {
        self
    }
    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }
    fn into_record(self) -> Self::Record {
        EmptyRecord
    }
    fn load_record(self, _: Self::Record) -> Self {
        self
    }
    fn num_params(&self) -> usize {
        0
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for Param<Flag> {
    type InnerModule = Self;
    fn valid(&self) -> Self {
        let mut flag = self.clone().map(|flag| flag.with(false));
        flag.require_grad = self.require_grad;
        flag
    }
    fn from_inner(flag: Self) -> Self {
        let enabled = flag.require_grad;
        flag.map(|flag| flag.with(enabled))
    }
}

impl<B: AutodiffBackend> HasAutodiffModule<B> for Param<Flag> {
    type TrainModule = Self;
}

impl ModuleDisplayDefault for Param<Flag> {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add_formatted(&alloc::format!("{}", self.val()))
            .optional()
    }
}
impl ModuleDisplay for Param<Flag> {}
