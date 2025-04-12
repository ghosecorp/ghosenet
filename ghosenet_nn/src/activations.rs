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