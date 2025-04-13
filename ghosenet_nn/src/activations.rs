use ghosenet_tensor::tensor::Tensor;
use crate::Module;
use ghosenet_tensor::ops::{exp, div, add, mul};
use std::f32;

pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut result = input.clone();
        for i in 0..result.data.len() {
            result.data[i] = result.data[i].max(0.0);
        }
        result
    }
    
    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
    
    fn zero_grad(&mut self) {
        // No parameters to zero out
    }
}

pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut result = input.clone();
        for i in 0..result.data.len() {
            result.data[i] = 1.0 / (1.0 + (-result.data[i]).exp());
        }
        result
    }
    
    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
    
    fn zero_grad(&mut self) {
        // No parameters to zero out
    }
}

pub struct Tanh;

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut result = input.clone();
        for i in 0..result.data.len() {
            result.data[i] = result.data[i].tanh();
        }
        result
    }
    
    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
    
    fn zero_grad(&mut self) {
        // No parameters to zero out
    }
}

pub struct LeakyReLU {
    pub negative_slope: f32,
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut result = input.clone();
        for i in 0..result.data.len() {
            result.data[i] = if result.data[i] > 0.0 {
                result.data[i]
            } else {
                self.negative_slope * result.data[i]
            };
        }
        result
    }
    
    fn parameters(&self) -> Vec<Tensor> { Vec::new() }
    fn zero_grad(&mut self) {}
}

pub struct Softmax;

impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Tensor {
        let max_val = input.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = input.data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let data = exp_vals.iter().map(|&x| x / sum_exp).collect();
        Tensor::new(data, input.shape.clone(), input.requires_grad)
    }
    
    fn parameters(&self) -> Vec<Tensor> { Vec::new() }
    fn zero_grad(&mut self) {}
}

pub struct Swish;

impl Module for Swish {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut result = input.clone();
        for i in 0..result.data.len() {
            let x = result.data[i];
            result.data[i] = x / (1.0 + (-x).exp()); // x * sigmoid(x)
        }
        result
    }
    
    fn parameters(&self) -> Vec<Tensor> { Vec::new() }
    fn zero_grad(&mut self) {}
}