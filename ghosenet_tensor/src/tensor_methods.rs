// tensor_methods.r

use super::Tensor;
// use std::iter::repeat;
use std::ops::{Add, Mul, Index, IndexMut};
use crate::ops::{add, mul};

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "Shape mismatch");
        let data_len = data.len();
        Tensor {
            data,
            shape,
            grad: if requires_grad { Some(vec![0.0; data_len]) } else { None },
            requires_grad,
            op: None,
            grad_fn: None,
        }
    }
    
    pub fn zeros(shape: Vec<usize>, requires_grad: bool) -> Self {
        let size = shape.iter().product::<usize>();
        Tensor {
            data: vec![0.0; size],
            shape,
            grad: if requires_grad { Some(vec![0.0; size]) } else { None },
            requires_grad,
            grad_fn: None,
        }
    }
    
    pub fn get(&self, index: usize) -> f32 {
        self.data[index]
    }
    
    pub fn set(&mut self, index: usize, value: f32) {
        self.data[index] = value;
    }
    
    pub fn get_at(&self, indices: &[usize]) -> f32 {
        let flat_index = self.calculate_flat_index(indices);
        self.data[flat_index]
    }
    
    pub fn set_at(&mut self, indices: &[usize], value: f32) {
        let flat_index = self.calculate_flat_index(indices);
        self.data[flat_index] = value;
    }
    
    pub fn calculate_flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "Dimension mismatch");
        let mut index = 0;
        let mut stride = 1;
        for (i, &dim) in self.shape.iter().rev().enumerate() {
            index += indices[self.shape.len() - 1 - i] * stride;
            stride *= dim;
        }
        index
    }
    
    pub fn calc_multi_index(&self, flat_index: usize) -> Vec<usize> {
        let mut indices = vec![0; self.shape.len()];
        let mut remainder = flat_index;
        for (i, dim) in self.shape.iter().rev().enumerate() {
            let dim = *dim;
            indices[self.shape.len() - 1 - i] = remainder % dim;
            remainder /= dim;
        }
        indices
    }
    
    pub fn iter(&self) -> std::slice::Iter<f32> {
        self.data.iter()
    }
    
    pub fn backward(&mut self) {
        // Initialize gradient if it doesn't exist with ones for output of computation
        if self.grad.is_none() && self.requires_grad {
            let size = self.data.len();
            // Start with gradient of 1.0 at the output
            self.grad = Some(vec![1.0; size]);
        }

        // If we have a gradient function and gradients are required
        if let Some(node) = &self.grad_fn {
            let grad_output = self.grad.as_ref().expect("Gradient is None");

            // Get input tensors - Fix the borrowing issue
            let input_tensors: Vec<Tensor> = node.inputs.iter()
                .map(|t| t.borrow().clone()) // Clone here to avoid lifetime issues
                .collect();
            
            // Create references to the cloned tensors
            let input_refs: Vec<&Tensor> = input_tensors.iter().collect();

            // Compute gradients for each input
            let grads = node.operation.backward(grad_output, &input_refs);

            // Update gradients of input tensors
            for (i, maybe_grad) in grads.iter().enumerate() {
                if let Some(grad) = maybe_grad {
                    let mut input = node.inputs[i].borrow_mut();
                    if input.requires_grad {
                        if let Some(existing_grad) = &mut input.grad {
                            // Accumulate gradients
                            for (j, g) in grad.iter().enumerate() {
                                existing_grad[j] += g;
                            }
                        } else {
                            input.grad = Some(grad.clone());
                        }

                        // Recursively backpropagate
                        drop(input); // Release borrow before recursive call
                        node.inputs[i].borrow_mut().backward();
                    }
                }
            }
        }
    }

    // Utility to reset the gradients
    pub fn zero_grad(&mut self) {
        if let Some(grad) = &mut self.grad {
            for g in grad.iter_mut() {
                *g = 0.0;
            }
        }
    }

    // Serialization: Save Tensor to file
    pub fn save_to_file(&self, path: &str) -> Result<(), std::io::Error> {
        use std::fs::File;
        use std::io::Write;
        let serialized = serde_json::to_string(self)?;
        let mut file = File::create(path)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    // Deserialize Tensor from file
    pub fn load_from_file(path: &str) -> Result<Self, std::io::Error> {
        use std::fs::File;
        use std::io::Read;
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let tensor: Tensor = serde_json::from_str(&contents)?;
        Ok(tensor)
    }
}

impl Index<usize> for Tensor {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        add(&self, &rhs)
    }
}

impl Mul for Tensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        mul(&self, &rhs)
    }
}