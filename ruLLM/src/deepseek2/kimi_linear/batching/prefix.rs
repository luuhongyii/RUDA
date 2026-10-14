use super::*;

impl<R, F, I, BT> PrefixBatchModel<DeviceBackend<R, F, I, BT>> for KimiLinearModel<DeviceBackend<R, F, I, BT>>
where R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps, F: FloatElement, I: IntElement, BT: BoolElement {
    type Prefix = KimiLinearBatchPrefix<DeviceBackend<R, F, I, BT>>;
    fn capture_prefix(&self, cache: &Self::Cache, id: RequestId) -> Result<Self::Prefix, GenerationError> {
        let recurrent = cache.recurrent.get(&id).ok_or_else(|| error("unknown Kimi Linear prefix request"))?.clone();
        self.validate_recurrent(&recurrent, true)?;
        let dense = cache.dense.capture_prefix(id)?;
        dense.validate(&self.dense_geometry(), self.config.model_max_length)?;
        Ok(Self::Prefix {dense, recurrent})
    }
    fn prefix_position(&self, prefix: &Self::Prefix) -> usize { prefix.dense.position() }
    fn prefix_bytes(&self, prefix: &Self::Prefix) -> usize {
        prefix.recurrent.iter().fold(prefix.dense.bytes(), |bytes, state| {
            let conv = state.conv.as_ref().map_or(0, |conv| conv.iter().fold(0usize, |n, x|
                n.saturating_add(x.dims().iter().product::<usize>().saturating_mul(x.dtype().size()))));
            let recurrent = state.recurrent.as_ref().map_or(0, |x| x.dims().iter().product::<usize>().saturating_mul(x.dtype().size()));
            bytes.saturating_add(conv).saturating_add(recurrent)
        })
    }
    fn restore_prefix(&self, cache: &mut Self::Cache, id: RequestId, table: &[KvPageId], prefix: &Self::Prefix) -> Result<(), GenerationError> {
        prefix.dense.validate(&self.dense_geometry(), self.config.model_max_length)?;
        self.validate_recurrent(&prefix.recurrent, true)?;
        if cache.recurrent.contains_key(&id) { return Err(error("Kimi Linear prefix destination already exists")); }
        cache.dense.restore_prefix(id, table, &prefix.dense)?;
        cache.recurrent.insert(id, prefix.recurrent.clone()); Ok(())
    }
    fn encode_prefix(&self, prefix: &Self::Prefix, model_id: &str) -> Result<Vec<u8>, GenerationError> {
        if model_id.is_empty() { return Err(error("Kimi Linear prefix requires model identity")); }
        prefix.dense.validate(&self.dense_geometry(), self.config.model_max_length)?;
        self.validate_recurrent(&prefix.recurrent, true)?;
        let mut states = Vec::with_capacity(prefix.recurrent.len());
        for state in &prefix.recurrent {
            let mut data = state.conv.as_ref().unwrap().iter().map(|x| x.clone().try_into_data().map_err(error)).collect::<Result<Vec<_>, _>>()?;
            data.push(state.recurrent.as_ref().unwrap().clone().try_into_data().map_err(error)?); states.push(data);
        }
        rmp_serde::to_vec(&(1u32, model_id, prefix.dense.encode()?, states)).map_err(error)
    }
    fn decode_prefix(&self, bytes: &[u8], model_id: &str, device: &R::Device) -> Result<Self::Prefix, GenerationError> {
        let (version, identity, dense, states): (u32, String, Vec<u8>, Vec<Vec<TensorData>>) = rmp_serde::from_slice(bytes).map_err(error)?;
        if version != 1 || model_id.is_empty() || identity != model_id || *device != self.embedding.device()
            || states.len() != self.config.num_hidden_layers - self.config.mla_layers() { return Err(error("Kimi Linear prefix identity/device/layout differs")); }
        if let Some(c) = &self.config.linear_attn_config {
            let conv = [1, c.num_heads * c.head_dim, c.short_conv_kernel_size]; let state = [1, c.num_heads, c.head_dim, c.head_dim];
            for data in &states {
                if data.len() != 4 { return Err(error("incomplete Kimi Linear prefix recurrent state")); }
                for (i, data) in data.iter().enumerate() {
                    let (shape, dtype) = if i == 3 {(&state[..], DType::F32)} else {(&conv[..], self.embedding.dtype())};
                    if data.shape.as_slice() != shape || data.dtype != dtype
                        || shape.iter().try_fold(dtype.size(), |n, &d| n.checked_mul(d)) != Some(data.as_bytes().len()) {
                        return Err(error("invalid Kimi Linear prefix native tensor payload"));
                    }
                }
            }
        }
        let dense = DenseBatchPrefix::decode(&dense, &self.dense_geometry(), self.config.model_max_length, device)?;
        let recurrent = states.into_iter().map(|data| {
            let mut data = data.into_iter();
            let conv = std::array::from_fn(|_| {let value = data.next().unwrap(); let dtype = value.dtype; Tensor::from_data(value, (device, dtype))});
            kda::State {conv: Some(conv), recurrent: Some(Tensor::from_data(data.next().unwrap(), (device, DType::F32)))}
        }).collect();
        Ok(Self::Prefix {dense, recurrent})
    }
}
