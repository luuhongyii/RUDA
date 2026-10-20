use ruda_tensor::{
    Backend, ExecutionError,
    ops::{TransactionOps, TransactionPrimitive},
};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy};

impl<B: Backend, C: CheckpointStrategy> TransactionOps<Self> for Autodiff<B, C> {
    async fn tr_execute(
        transaction: TransactionPrimitive<Self>,
    ) -> Result<ruda_tensor::ops::TransactionPrimitiveData, ExecutionError> {
        B::tr_execute(TransactionPrimitive::new(
            transaction
                .read_floats
                .into_iter()
                .map(|t| t.primitive)
                .collect(),
            transaction
                .read_qfloats
                .into_iter()
                .map(|t| t.primitive)
                .collect(),
            transaction.read_ints.into_iter().map(|t| t.primitive).collect(),
            transaction.read_bools.into_iter().map(|t| t.primitive).collect(),
        ))
        .await
    }
}
