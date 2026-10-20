use ruda_kernel::dsl::server::ExecutionMode;

use crate::memory::arguments::MlirData;
use crate::execution::jit::MlirEngine;
use std::sync::mpsc;

pub use super::barrier::sync_ruda;

pub enum Message {
    ComputeTask(ComputeTask),
    EndTask(mpsc::Sender<()>),
}

pub struct ComputeTask {
    pub mlir_engine: MlirEngine,
    pub mlir_data: MlirData,
    pub unit_pos: [u32; 3],
    pub kind: ExecutionMode,
}

impl ComputeTask {
    pub fn compute(mut self) {
        self.mlir_data.push_builtin();
        self.mlir_data.builtin.set_unit_pos(self.unit_pos);
        unsafe {
            self.mlir_engine.run_kernel(&mut self.mlir_data);
        }
    }
}
