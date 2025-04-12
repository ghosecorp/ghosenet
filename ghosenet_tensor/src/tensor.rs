// tensor.ts
use std::ops::{Add, Mul, Index, IndexMut};
// use crate::ops::{add, mul, exp, log, sum, mean};
use crate::ops::{add, mul};

use serde::{Serialize, Deserialize};
// use serde_json::{to_string, from_str};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub grad: Option<Vec<f32>>,
    pub requires_grad: bool,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "Shape mismatch");
        Tensor {
            data,
            shape,
            grad: None,
            requires_grad,
        }
    }

    pub fn zeros(shape: Vec<usize>, requires_grad: bool) -> Self {
        let size = shape.iter().product::<usize>();
        Tensor {
            data: vec![0.0; size],
            shape,
            grad: None,
            requires_grad,
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

    fn calculate_flat_index(&self, indices: &[usize]) -> usize {
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
        if self.requires_grad {
            if self.grad.is_none() {
                // Start with gradient of 1.0 for the output tensor
                let mut grad = vec![0.0; self.data.len()];
                
                // For scalar output, set gradient to 1.0
                // For multi-element tensor, this could be modified based on needs
                if self.data.len() == 1 {
                    grad[0] = 1.0;
                } else {
                    // For non-scalar tensors, we need the calling code to set
                    // the initial gradient, or we can set it to 1.0 for each element
                    for i in 0..grad.len() {
                        grad[i] = 1.0;
                    }
                }
                
                self.grad = Some(grad);
            }
            
            // Here's where we would propagate the gradient backward through
            // the computation graph, but we need to track operations first
            
            // For a complete autodiff system, you would need to:
            // 1. Track operations that created this tensor
            // 2. For each operation, compute gradients with respect to inputs
            // 3. Accumulate these gradients in the input tensors
            // 4. Call backward recursively on input tensors
        }
    }

    // Utility to reset the gradients
    pub fn zero_grad(&mut self) {
        self.grad = None;
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