use super::{BiLstmConfig, GateController, LstmConfig, LstmState};
use super::lstm::{BiLstmRef, LstmRef};
use crate::activation::Activation;
use ruda_model::module::Module;
use ruda_model::tensor::{Tensor, backend::Backend};

/// LSTM with a coupled forget gate (`f = 1 - i`) and only three stored gates.
#[derive(Module, Debug)]
pub struct CoupledLstm<B: Backend> {
    /// Input gate.
    pub input_gate: GateController<B>,
    /// Output gate.
    pub output_gate: GateController<B>,
    /// Candidate cell gate.
    pub cell_gate: GateController<B>,
    /// Hidden state size.
    pub d_hidden: usize,
    /// Whether the input and output layouts are batch-first.
    pub batch_first: bool,
    /// Whether to process the sequence in reverse.
    pub reverse: bool,
    /// Optional cell state clipping threshold.
    pub clip: Option<f64>,
    /// Gate activation.
    pub gate_activation: Activation<B>,
    /// Candidate cell activation.
    pub cell_activation: Activation<B>,
    /// Hidden output activation.
    pub hidden_activation: Activation<B>,
}

impl LstmConfig {
    /// Initialize the three-gate coupled variant, without allocating forget parameters.
    pub fn init_coupled<B: Backend>(&self, device: &B::Device) -> CoupledLstm<B> {
        let new_gate = || GateController::new(
            self.d_input, self.d_hidden, self.bias, self.initializer.clone(), device,
        );
        CoupledLstm {
            input_gate: new_gate(),
            output_gate: new_gate(),
            cell_gate: new_gate(),
            d_hidden: self.d_hidden,
            batch_first: self.batch_first,
            reverse: self.reverse,
            clip: self.clip,
            gate_activation: self.gate_activation.init(device),
            cell_activation: self.cell_activation.init(device),
            hidden_activation: self.hidden_activation.init(device),
        }
    }
}

impl<B: Backend> CoupledLstm<B> {
    fn as_ref(&self) -> LstmRef<'_, B> {
        LstmRef {
            input_gate: &self.input_gate,
            forget_gate: &self.input_gate,
            output_gate: &self.output_gate,
            cell_gate: &self.cell_gate,
            d_hidden: self.d_hidden,
            batch_first: self.batch_first,
            reverse: self.reverse,
            clip: self.clip,
            input_forget: true,
            gate_activation: &self.gate_activation,
            cell_activation: &self.cell_activation,
            hidden_activation: &self.hidden_activation,
        }
    }

    /// Return the sequence outputs and final cell/hidden states, using the LSTM layout contract.
    pub fn forward(&self, input: Tensor<B, 3>, state: Option<LstmState<B, 2>>)
        -> (Tensor<B, 3>, LstmState<B, 2>) {
        self.as_ref().forward(input, state)
    }
}

/// Bidirectional coupled LSTM with three stored gates per direction.
#[derive(Module, Debug)]
pub struct CoupledBiLstm<B: Backend> {
    /// Forward-direction module.
    pub forward: CoupledLstm<B>,
    /// Reverse-direction module.
    pub reverse: CoupledLstm<B>,
    /// Hidden state size per direction.
    pub d_hidden: usize,
    /// Whether the input and output layouts are batch-first.
    pub batch_first: bool,
}

impl BiLstmConfig {
    /// Initialize both directions without allocating forget parameters.
    pub fn init_coupled<B: Backend>(&self, device: &B::Device) -> CoupledBiLstm<B> {
        let config = LstmConfig::new(self.d_input, self.d_hidden, self.bias)
            .with_initializer(self.initializer.clone())
            .with_batch_first(true)
            .with_clip(self.clip)
            .with_gate_activation(self.gate_activation.clone())
            .with_cell_activation(self.cell_activation.clone())
            .with_hidden_activation(self.hidden_activation.clone());
        CoupledBiLstm {
            forward: config.init_coupled(device),
            reverse: config.init_coupled(device),
            d_hidden: self.d_hidden,
            batch_first: self.batch_first,
        }
    }
}

impl<B: Backend> CoupledBiLstm<B> {
    /// Return concatenated direction outputs and final states, using the BiLSTM layout contract.
    pub fn forward(&self, input: Tensor<B, 3>, state: Option<LstmState<B, 3>>)
        -> (Tensor<B, 3>, LstmState<B, 3>) {
        BiLstmRef {
            forward: self.forward.as_ref(),
            reverse: self.reverse.as_ref(),
            d_hidden: self.d_hidden,
            batch_first: self.batch_first,
        }.forward(input, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Initializer, TestBackend};
    use ruda_model::tensor::{TensorData, Tolerance};

    #[test]
    fn coupled_parameter_counts() {
        let device = Default::default();
        let uni = LstmConfig::new(2, 4, true).init_coupled::<TestBackend>(&device);
        let bi = BiLstmConfig::new(2, 4, true).init_coupled::<TestBackend>(&device);
        assert_eq!(uni.num_params(), 3 * (2 * 4 + 4 * 4 + 2 * 4));
        assert_eq!(bi.num_params(), 2 * uni.num_params());
    }

    #[test]
    fn coupled_forward_initializes_owned_parameters_once() {
        let device = Default::default();
        TestBackend::seed(&device, 17);
        let input = Tensor::<TestBackend, 3>::ones([1, 3, 2], &device);
        let uni = LstmConfig::new(2, 4, true).init_coupled::<TestBackend>(&device);
        assert!(!uni.input_gate.input_transform.weight.is_initialized());
        let first = uni.forward(input.clone(), None).0;
        assert!(uni.input_gate.input_transform.weight.is_initialized());
        assert!(uni.cell_gate.hidden_transform.bias.as_ref().unwrap().is_initialized());
        assert_eq!(first.into_data(), uni.forward(input.clone(), None).0.into_data());

        let bi = BiLstmConfig::new(2, 4, true).init_coupled::<TestBackend>(&device);
        let first = bi.forward(input.clone(), None).0;
        assert!(bi.forward.input_gate.input_transform.weight.is_initialized());
        assert!(bi.reverse.input_gate.input_transform.weight.is_initialized());
        assert_eq!(first.into_data(), bi.forward(input, None).0.into_data());
    }

    #[test]
    fn coupled_gradients_reach_owned_parameters() {
        use crate::TestAutodiffBackend;
        let device = Default::default();
        let module = LstmConfig::new(2, 4, true).init_coupled::<TestAutodiffBackend>(&device);
        let input = Tensor::<TestAutodiffBackend, 3>::ones([1, 3, 2], &device);
        let grads = module.forward(input, None).0.sum().backward();
        for gate in [&module.input_gate, &module.output_gate, &module.cell_gate] {
            assert!(gate.input_transform.weight.val().grad(&grads).is_some());
            assert!(gate.hidden_transform.weight.val().grad(&grads).is_some());
        }
    }

    fn reference(reverse: bool) -> ([f32; 3], f32, f32) {
        let x = [0.2f64, -0.1, 0.4];
        let mut outputs = [0.0; 3];
        let (mut cell, mut hidden) = (0.3f64, -0.2f64);
        for step in 0..3 {
            let t = if reverse { 2 - step } else { step };
            let sum = x[t] + hidden;
            let gate = 1.0 / (1.0 + (-sum).exp());
            cell = ((1.0 - gate) * cell + gate * sum.tanh()).clamp(-0.25, 0.25);
            hidden = gate * cell.tanh();
            outputs[t] = hidden as f32;
        }
        (outputs, cell as f32, hidden as f32)
    }

    #[test]
    fn coupled_directions_layouts_and_initial_states() {
        let device = Default::default();
        for batch_first in [false, true] {
            let shape = if batch_first { [1, 3, 1] } else { [3, 1, 1] };
            let input = Tensor::<TestBackend, 3>::from_data(
                TensorData::new(alloc::vec![0.2f32, -0.1, 0.4], shape), &device,
            );
            for reverse in [false, true] {
                let module = LstmConfig::new(1, 1, false)
                    .with_initializer(Initializer::Ones)
                    .with_batch_first(batch_first).with_reverse(reverse).with_clip(Some(0.25))
                    .init_coupled::<TestBackend>(&device);
                let state = LstmState::new(Tensor::full([1, 1], 0.3, &device),
                    Tensor::full([1, 1], -0.2, &device));
                let (out, state) = module.forward(input.clone(), Some(state));
                let (expected, cell, hidden) = reference(reverse);
                out.into_data().assert_approx_eq::<f32>(&TensorData::new(expected.to_vec(), shape), Tolerance::absolute(1e-6));
                state.cell.into_data().assert_approx_eq::<f32>(&TensorData::from([[cell]]), Tolerance::absolute(1e-6));
                state.hidden.into_data().assert_approx_eq::<f32>(&TensorData::from([[hidden]]), Tolerance::absolute(1e-6));
            }
            let module = BiLstmConfig::new(1, 1, false)
                .with_initializer(Initializer::Ones).with_batch_first(batch_first).with_clip(Some(0.25))
                .init_coupled::<TestBackend>(&device);
            let state = LstmState::new(Tensor::full([2, 1, 1], 0.3, &device),
                Tensor::full([2, 1, 1], -0.2, &device));
            let (out, state) = module.forward(input, Some(state));
            let (forward, fc, fh) = reference(false);
            let (reverse, rc, rh) = reference(true);
            let expected: alloc::vec::Vec<_> = (0..3).flat_map(|t| [forward[t], reverse[t]]).collect();
            let out_shape = if batch_first { [1, 3, 2] } else { [3, 1, 2] };
            out.into_data().assert_approx_eq::<f32>(&TensorData::new(expected, out_shape), Tolerance::absolute(1e-6));
            state.cell.into_data().assert_approx_eq::<f32>(&TensorData::from([[[fc]], [[rc]]]), Tolerance::absolute(1e-6));
            state.hidden.into_data().assert_approx_eq::<f32>(&TensorData::from([[[fh]], [[rh]]]), Tolerance::absolute(1e-6));
        }
    }
}
