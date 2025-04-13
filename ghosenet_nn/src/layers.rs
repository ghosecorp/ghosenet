use ghosenet_tensor::tensor::{Tensor, OpType};
use ghosenet_tensor::ops::{matmul, add};
use crate::{Module, Parameter};
use crate::init::{xavier_uniform, zeros};

// Linear layer (fully connected)
pub struct Linear {
    pub in_features: usize,
    pub out_features: usize,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Initialize with Xavier uniform distribution
        let weight = xavier_uniform(&[out_features, in_features], true);
        
        let bias = if bias {
            // Bias shape should be [1, out_features] for broadcasting
            Some(zeros(&[1, out_features], true))
        } else {
            None
        };
        
        Linear {
            in_features,
            out_features,
            weight,
            bias,
        }
    }
    
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Ensure input has the right shape
        assert!(input.shape.len() >= 1 && input.shape.last().unwrap() == &self.in_features,
                "Expected input to have shape [..., {}], but got {:?}", 
                self.in_features, input.shape);
        
        let output = matmul(input, &self.weight.transpose());
        
        if let Some(ref bias) = self.bias {
            add(&output, bias)
        } else {
            output
        }
    }
    
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
    
    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        if let Some(ref mut bias) = self.bias {
            bias.zero_grad();
        }
    }
}

// Dropout layer
pub struct Dropout {
    pub p: f32,
    pub training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(p >= 0.0 && p < 1.0, "Dropout probability must be between 0 and 1");
        Dropout { p, training: true }
    }
    
    pub fn train(&mut self) {
        self.training = true;
    }
    
    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }
        
        // In a real implementation, we'd generate a random mask here
        // For now, we'll simulate it with a simplified approach
        let scale = 1.0 / (1.0 - self.p);
        let mut result = input.clone();
        
        // Apply dropout (this is a simplified implementation)
        // In practice, we'd use a random mask
        for i in 0..result.data.len() {
            if i % 4 != 0 { // Simple deterministic "random" for example purposes
                result.data[i] *= scale;
            } else {
                result.data[i] = 0.0;
            }
        }
        
        result
    }
    
    fn parameters(&self) -> Vec<Tensor> {
        // Dropout has no parameters
        Vec::new()
    }
    
    fn zero_grad(&mut self) {
        // Nothing to do
    }
}