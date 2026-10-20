use super::*;
use crate::{DispatchDevice, BackendTensor};
use ruda_tensor::DeviceOps;

#[derive(Default)]
struct Reads {
    floats: Vec<(usize, DispatchTensor)>,
    quantized: Vec<(usize, DispatchTensor)>,
    ints: Vec<(usize, DispatchTensor)>,
    bools: Vec<(usize, DispatchTensor)>,
}

impl TransactionOps<Self> for Dispatch {
    fn tr_execute(transaction: TransactionPrimitive<Self>) -> impl Future<Output = Result<TransactionPrimitiveData, ExecutionError>> + Send {
        let mut groups: Vec<(DispatchDevice, Reads)> = Vec::new();
        let sizes = [transaction.read_floats.len(), transaction.read_qfloats.len(), transaction.read_ints.len(), transaction.read_bools.len()];
        macro_rules! group {
            ($values:expr, $field:ident) => {
                for (index, tensor) in $values.into_iter().enumerate() {
                    let device = tensor.device().inner().clone();
                    let position = groups.iter().position(|(candidate, _)| candidate == &device)
                        .unwrap_or_else(|| { groups.push((device, Reads::default())); groups.len() - 1 });
                    groups[position].1.$field.push((index, tensor));
                }
            };
        }
        group!(transaction.read_floats, floats);
        group!(transaction.read_qfloats, quantized);
        group!(transaction.read_ints, ints);
        group!(transaction.read_bools, bools);
        let pending = groups.into_iter().map(|(device, reads)| {
            macro_rules! execute {
                ($B:ty, $variant:ident, $device:expr) => {{
                    let (float_indices, floats): (Vec<_>, Vec<_>) = reads.floats.into_iter()
                        .map(|(i, t)| (i, unpack!(native, $variant, Float, t, AutodiffContext::Disabled))).unzip();
                    let (quantized_indices, quantized): (Vec<_>, Vec<_>) = reads.quantized.into_iter()
                        .map(|(i, t)| (i, unpack!(native, $variant, Quantized, t, AutodiffContext::Disabled))).unzip();
                    let (int_indices, ints): (Vec<_>, Vec<_>) = reads.ints.into_iter()
                        .map(|(i, t)| (i, unpack!(native, $variant, Int, t, AutodiffContext::Disabled))).unzip();
                    let (bool_indices, bools): (Vec<_>, Vec<_>) = reads.bools.into_iter()
                        .map(|(i, t)| (i, unpack!(native, $variant, Bool, t, AutodiffContext::Disabled))).unzip();
                    let future = <$B>::tr_execute(TransactionPrimitive::new(floats, quantized, ints, bools));
                    let future: core::pin::Pin<Box<dyn Future<Output = Result<TransactionPrimitiveData, ExecutionError>> + Send>> = Box::pin(future);
                    ([float_indices, quantized_indices, int_indices, bool_indices], future)
                }};
            }
            native_dispatch!(&device, execute)
        }).collect::<Vec<_>>();
        async move {
            let mut output = sizes.map(|size| (0..size).map(|_| None).collect::<Vec<Option<TensorData>>>());
            for (indices, future) in pending {
                let data = future.await?;
                for ((indices, values), destination) in indices.into_iter()
                    .zip([data.read_floats, data.read_qfloats, data.read_ints, data.read_bools])
                    .zip(output.iter_mut()) {
                    assert_eq!(indices.len(), values.len(), "Backend returned an incomplete transaction");
                    for (index, value) in indices.into_iter().zip(values) { destination[index] = Some(value); }
                }
            }
            let [floats, quantized, ints, bools] = output.map(|values| values.into_iter()
                .map(|value| value.expect("Missing transaction result")).collect());
            Ok(TransactionPrimitiveData { read_floats: floats, read_qfloats: quantized, read_ints: ints, read_bools: bools })
        }
    }
}
