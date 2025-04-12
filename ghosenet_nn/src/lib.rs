use ghosenet_tensor::tensor::Tensor;
use std::collections::HashMap;

pub mod activations;
pub mod layers;
pub mod loss;
pub mod init;

// Module trait defines the interface for all neural network components
pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    fn zero_grad(&mut self);
}

// Sequential container to stack layers
pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut current = input.clone();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }
}

// Parameter wrapper to manage trainable parameters
pub struct Parameter {
    pub value: Tensor,
}

impl Parameter {
    pub fn new(tensor: Tensor) -> Self {
        assert!(tensor.requires_grad, "Parameters must have requires_grad=true");
        Parameter { value: tensor }
    }
}

#[cfg(test)]
mod tests;