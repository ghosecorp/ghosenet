// tensor.rs
use crate::ops::{add, mul, select};
use std::cell::RefCell;
use std::ops::{Add, Index, IndexMut, Mul};
use std::rc::Rc;

use serde::{Deserialize, Serialize};

// Define operation types for the computational graph
#[derive(Debug, Clone)]
pub enum OpType {
    Add,
    Mul,
    MatMul,
    Exp,
    Log,
    Sum,
    Mean,
    Div,
    Sub,
    Abs,
    Select,
    Input, // Placeholder for leaf tensors
}

pub struct OpNode {
    pub op_type: OpType,
    pub inputs: Vec<Rc<RefCell<Tensor>>>,
    pub backward: Option<Box<dyn Fn(&Tensor) -> ()>>,
}

// Manually implement Debug
impl std::fmt::Debug for OpNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpNode")
            .field("op_type", &self.op_type)
            .field("inputs", &self.inputs)
            .field(
                "backward",
                &if self.backward.is_some() {
                    "Some(Fn)"
                } else {
                    "None"
                },
            )
            .finish()
    }
}

// Manually implement Clone
impl Clone for OpNode {
    fn clone(&self) -> Self {
        OpNode {
            op_type: self.op_type.clone(),
            inputs: self.inputs.clone(),
            // We can't clone the function, so set it to None when cloning
            backward: None,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub grad: Option<Vec<f32>>,
    pub requires_grad: bool,

    #[serde(skip)]
    pub op: Option<OpNode>, // Operation that created this tensor
    #[serde(skip)]
    pub grad_fn: Option<Box<dyn Fn() -> ()>>, // Function to compute gradients
}

// Implement Debug manually:
impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("data", &self.data)
            .field("shape", &self.shape)
            .field("grad", &self.grad)
            .field("requires_grad", &self.requires_grad)
            .field("op", &self.op)
            .field(
                "grad_fn",
                &if self.grad_fn.is_some() {
                    "<function>"
                } else {
                    "None"
                },
            )
            .finish()
    }
}

// Implement Clone manually:
impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            grad: self.grad.clone(),
            requires_grad: self.requires_grad,
            op: self.op.clone(),
            grad_fn: None, // We cannot clone the function
        }
    }
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Shape mismatch"
        );
        let data_len = data.len();
        Tensor {
            data,
            shape,
            grad: if requires_grad {
                Some(vec![0.0; data_len])
            } else {
                None
            },
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
            grad: if requires_grad {
                Some(vec![0.0; size])
            } else {
                None
            },
            requires_grad,
            op: None,
            grad_fn: None,
        }
    }

    pub fn get(&self, index: usize) -> f32 {
        if index >= self.data.len() {
            panic!(
                "index out of bounds: the len is {} but the index is {}",
                self.data.len(),
                index
            );
        }
        self.data[index]
    }

    pub fn set(&mut self, index: usize, value: f32) {
        self.data[index] = value;
    }

    pub fn get_at(&self, indices: &[usize]) -> f32 {
        let flat_index = self.calculate_flat_index(indices);
        if flat_index >= self.data.len() {
            panic!(
                "index out of bounds: the len is {} but the index is {}",
                self.data.len(),
                flat_index
            );
        }
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

    // Add a method to accumulate gradients
    pub fn accumulate_grad(&mut self, grad: &[f32]) {
        if self.grad.is_none() {
            self.grad = Some(grad.to_vec());
        } else if let Some(ref mut self_grad) = self.grad {
            assert_eq!(self_grad.len(), grad.len(), "Gradient size mismatch");
            for i in 0..self_grad.len() {
                self_grad[i] += grad[i];
            }
        }
    }

    pub fn backward(&mut self) {
        // Initialize gradient for output tensor
        if self.requires_grad {
            if self.grad.is_none() {
                self.grad = Some(vec![1.0; self.data.len()]);
            }

            // Set initial gradient to 1.0 for scalar output or for the element being backpropagated
            let grad = self.grad.as_mut().unwrap();
            if self.data.len() == 1 {
                grad[0] = 1.0;
            } else {
                // For non-scalar tensors, set all gradients to 1.0
                // In practice, you might want to be more specific about which gradient to set
                for i in 0..grad.len() {
                    grad[i] = 1.0;
                }
            }

            // Call the gradient function if it exists
            if let Some(ref grad_fn) = self.grad_fn {
                grad_fn();
            }
        }
    }

    // Utility to reset the gradients
    pub fn zero_grad(&mut self) {
        if let Some(ref mut grad) = self.grad {
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

    // Inside impl Tensor
    // pub fn randn(shape: Vec<usize>, requires_grad: bool) -> Self {
    //     use rand_distr::{Distribution, Normal};

    //     let total_elems = shape.iter().product();
    //     let normal = Normal::new(0.0, 1.0).unwrap();
    //     let data: Vec<f32> = (0..total_elems)
    //         .map(|_| normal.sample(&mut rand::thread_rng()) as f32)
    //         .collect();

    //     Tensor::new(data, shape, requires_grad)
    // }

    pub fn transpose(&self) -> Self {
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut transposed_data = vec![0.0; self.data.len()];

        for i in 0..rows {
            for j in 0..cols {
                transposed_data[j * rows + i] = self.data[i * cols + j];
            }
        }

        Tensor::new(transposed_data, vec![cols, rows], self.requires_grad)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn abs(&self) -> Tensor {
        let result_data: Vec<f32> = self.data.iter().map(|&x| x.abs()).collect();
        let mut result = Tensor::new(result_data, self.shape.clone(), self.requires_grad);

        if self.requires_grad {
            let input = Rc::new(RefCell::new(self.clone()));
            let backward_input = Rc::clone(&input);

            result.op = Some(OpNode {
                op_type: OpType::Abs,
                inputs: vec![Rc::clone(&input)],
                backward: Some(Box::new(move |grad: &Tensor| {
                    let input_ref = backward_input.borrow();
                    let sign_data: Vec<f32> = input_ref.data.iter()
                        .map(|&x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 })
                        .collect();
                    let sign_tensor = Tensor::new(sign_data, input_ref.shape.clone(), false);
                    let local_grad = mul(grad, &sign_tensor);

                    drop(input_ref); // Drop immutable borrow

                    // Mutate parent's gradient in place
                    let mut input_mut = backward_input.borrow_mut();
                    if let Some(ref mut g) = input_mut.grad {
                        for (i, val) in local_grad.data.iter().enumerate() {
                            g[i] += val;
                        }
                    } else {
                        input_mut.grad = Some(local_grad.data.clone());
                    }
                })),
            });
        }

        result
    }

    pub fn less_than_scalar(&self, scalar: f32) -> Tensor {
        let mask_data: Vec<f32> = self.data.iter().map(|&x| if x < scalar { 1.0 } else { 0.0 }).collect();
        Tensor::new(mask_data, self.shape.clone(), false)
    }

    pub fn select(&self, if_true: &Tensor, if_false: &Tensor) -> Tensor {
        select(self, if_true, if_false)
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
