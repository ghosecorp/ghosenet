// autograd.rs
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;
use crate::tensor::Tensor;
use crate::ops::{add, sub, mul, div};

// Operation enum for tracking computational graph
#[derive(Debug, Clone)]
pub enum Operation {
    Add(Rc<Variable>, Rc<Variable>),
    Sub(Rc<Variable>, Rc<Variable>),
    Mul(Rc<Variable>, Rc<Variable>),
    Div(Rc<Variable>, Rc<Variable>),
    Pow(Rc<Variable>, f32),
    Tanh(Rc<Variable>),
    Relu(Rc<Variable>),
    Sigmoid(Rc<Variable>),
    None,
}

// Variable struct that wraps a Tensor for autograd
#[derive(Debug, Clone)]
pub struct Variable {
    pub data: Tensor,
    pub grad: Option<Box<Tensor>>,
    pub requires_grad: bool,
    pub operation: Operation,
    pub id: usize,
}

// Global counter for variable IDs
static mut NEXT_ID: usize = 0;

fn get_next_id() -> usize {
    unsafe {
        let id = NEXT_ID;
        NEXT_ID += 1;
        id
    }
}

impl Variable {
    pub fn new(tensor: Tensor) -> Self {
        let requires_grad = tensor.grad.is_some();

        Variable {
            grad: if requires_grad {
                Some(Box::new(Tensor::zeros(tensor.shape.clone())))
            } else {
                None
            },
            requires_grad,
            operation: Operation::None,
            id: get_next_id(),
            data: tensor,
        }
    }

    pub fn from_scalar(value: f32) -> Self {
        let tensor = Tensor::new(vec![value], vec![1]);
        Self::new(tensor)
    }

    // Mark variable to track gradients
    pub fn requires_grad(mut self) -> Self {
        self.requires_grad = true;
        if self.grad.is_none() {
            self.grad = Some(Box::new(Tensor::zeros(self.data.shape.clone())));
        }
        self
    }

    pub fn zero_grad(&mut self) {
        if let Some(g) = self.grad.as_mut() {
            for val in &mut g.data {
                *val = 0.0;
            }
        }
    }

    pub fn step(&mut self, learning_rate: f32) {
        if !self.requires_grad {
            return;
        }

        if let Some(grad) = &self.grad {
            for (data_val, grad_val) in self.data.data.iter_mut().zip(grad.data.iter()) {
                *data_val -= learning_rate * grad_val;
            }
        }
    }
}

// Overloaded forward ops
impl std::ops::Add for Rc<Variable> {
    type Output = Rc<Variable>;

    fn add(self, other: Rc<Variable>) -> Self::Output {
        let result_tensor = add(&self.data, &other.data);
        let requires_grad = self.requires_grad || other.requires_grad;

        let mut result = Variable::new(result_tensor);
        result.requires_grad = requires_grad;

        if requires_grad {
            result.operation = Operation::Add(self.clone(), other.clone());
            result.grad = Some(Box::new(Tensor::zeros(result.data.shape.clone())));
        }

        Rc::new(result)
    }
}

impl std::ops::Sub for Rc<Variable> {
    type Output = Rc<Variable>;

    fn sub(self, other: Rc<Variable>) -> Self::Output {
        let result_tensor = sub(&self.data, &other.data);
        let requires_grad = self.requires_grad || other.requires_grad;

        let mut result = Variable::new(result_tensor);
        result.requires_grad = requires_grad;

        if requires_grad {
            result.operation = Operation::Sub(self.clone(), other.clone());
            result.grad = Some(Box::new(Tensor::zeros(result.data.shape.clone())));
        }

        Rc::new(result)
    }
}

impl std::ops::Mul for Rc<Variable> {
    type Output = Rc<Variable>;

    fn mul(self, other: Rc<Variable>) -> Self::Output {
        let result_tensor = mul(&self.data, &other.data);
        let requires_grad = self.requires_grad || other.requires_grad;

        let mut result = Variable::new(result_tensor);
        result.requires_grad = requires_grad;

        if requires_grad {
            result.operation = Operation::Mul(self.clone(), other.clone());
            result.grad = Some(Box::new(Tensor::zeros(result.data.shape.clone())));
        }

        Rc::new(result)
    }
}

impl std::ops::Div for Rc<Variable> {
    type Output = Rc<Variable>;

    fn div(self, other: Rc<Variable>) -> Self::Output {
        let result_tensor = div(&self.data, &other.data);
        let requires_grad = self.requires_grad || other.requires_grad;

        let mut result = Variable::new(result_tensor);
        result.requires_grad = requires_grad;

        if requires_grad {
            result.operation = Operation::Div(self.clone(), other.clone());
            result.grad = Some(Box::new(Tensor::zeros(result.data.shape.clone())));
        }

        Rc::new(result)
    }
}


// Backward pass for computing gradients
pub fn backward(var: Rc<Variable>, grad_output: Option<Tensor>) {
    // Build topological sort of the computation graph
    let mut topo = Vec::new();
    let mut visited = HashSet::new();
    
    fn build_topo(var: &Rc<Variable>, topo: &mut Vec<Rc<Variable>>, visited: &mut HashSet<usize>) {
        if !visited.contains(&var.id) {
            visited.insert(var.id);
            
            match &var.operation {
                Operation::Add(a, b) | Operation::Sub(a, b) | 
                Operation::Mul(a, b) | Operation::Div(a, b) => {
                    build_topo(a, topo, visited);
                    build_topo(b, topo, visited);
                }
                Operation::Pow(a, _) | Operation::Tanh(a) | 
                Operation::Relu(a) | Operation::Sigmoid(a) => {
                    build_topo(a, topo, visited);
                }
                Operation::None => {}
            }
            
            topo.push(var.clone());
        }
    }
    
    build_topo(&var, &mut topo, &mut visited);
    
    // Initialize gradient
    if let Some(grad) = var.grad.as_ref() {
        if let Some(grad_output) = grad_output {
            // Use provided gradient
            assert_eq!(grad.shape, grad_output.shape, "Gradient shape mismatch");
            for (i, val) in grad_output.data.iter().enumerate() {
                let mut_var = unsafe { &mut *(Rc::as_ptr(&var) as *mut Variable) };
                if let Some(grad) = &mut mut_var.grad {
                    grad.data[i] = *val;
                }
            }
        } else {
            // Default to ones for scalar outputs
            let mut_var = unsafe { &mut *(Rc::as_ptr(&var) as *mut Variable) };
            if let Some(grad) = &mut mut_var.grad {
                if grad.data.len() == 1 {
                    grad.data[0] = 1.0;
                }
            }
        }
    }
    
    // Backpropagate
    for var in topo.iter().rev() {
        match &var.operation {
            Operation::Add(a, b) => {
                if a.requires_grad {
                    let mut a_mut = unsafe { &mut *(Rc::as_ptr(a) as *mut Variable) };
                    if let (Some(var_grad), Some(a_grad)) = (var.grad.as_ref(), &mut a_mut.grad) {
                        for (i, val) in var_grad.data.iter().enumerate() {
                            a_grad.data[i] += val;
                        }
                    }
                }
                
                if b.requires_grad {
                    let mut b_mut = unsafe { &mut *(Rc::as_ptr(b) as *mut Variable) };
                    if let (Some(var_grad), Some(b_grad)) = (var.grad.as_ref(), &mut b_mut.grad) {
                        for (i, val) in var_grad.data.iter().enumerate() {
                            b_grad.data[i] += val;
                        }
                    }
                }
            },
            Operation::Sub(a, b) => {
                if a.requires_grad {
                    let mut a_mut = unsafe { &mut *(Rc::as_ptr(a) as *mut Variable) };
                    if let (Some(var_grad), Some(a_grad)) = (var.grad.as_ref(), &mut a_mut.grad) {
                        for (i, val) in var_grad.data.iter().enumerate() {
                            a_grad.data[i] += val;
                        }
                    }
                }
                
                if b.requires_grad {
                    let mut b_mut = unsafe { &mut *(Rc::as_ptr(b) as *mut Variable) };
                    if let (Some(var_grad), Some(b_grad)) = (var.grad.as_ref(), &mut b_mut.grad) {
                        for (i, val) in var_grad.data.iter().enumerate() {
                            b_grad.data[i] -= val; // Subtraction: gradient flips sign
                        }
                    }
                }
            },
            Operation::Mul(a, b) => {
                if a.requires_grad {
                    let mut a_mut = unsafe { &mut *(Rc::as_ptr(a) as *mut Variable) };
                    if let (Some(var_grad), Some(a_grad)) = (var.grad.as_ref(), &mut a_mut.grad) {
                        for i in 0..var_grad.data.len() {
                            let indices = var_grad.calc_multi_index(i);
                            let b_indices = map_indices_for_broadcast(&indices, &b.data.shape);
                            a_grad.data[i] += var_grad.data[i] * b.data.get_at(&b_indices);
                        }
                    }
                }
                
                if b.requires_grad {
                    let mut b_mut = unsafe { &mut *(Rc::as_ptr(b) as *mut Variable) };
                    if let (Some(var_grad), Some(b_grad)) = (var.grad.as_ref(), &mut b_mut.grad) {
                        for i in 0..var_grad.data.len() {
                            let indices = var_grad.calc_multi_index(i);
                            let a_indices = map_indices_for_broadcast(&indices, &a.data.shape);
                            b_grad.data[i] += var_grad.data[i] * a.data.get_at(&a_indices);
                        }
                    }
                }
            },
            Operation::Div(a, b) => {
                if a.requires_grad {
                    let mut a_mut = unsafe { &mut *(Rc::as_ptr(a) as *mut Variable) };
                    if let (Some(var_grad), Some(a_grad)) = (var.grad.as_ref(), &mut a_mut.grad) {
                        for i in 0..var_grad.data.len() {
                            let indices = var_grad.calc_multi_index(i);
                            let b_indices = map_indices_for_broadcast(&indices, &b.data.shape);
                            a_grad.data[i] += var_grad.data[i] / b.data.get_at(&b_indices);
                        }
                    }
                }
                
                if b.requires_grad {
                    let mut b_mut = unsafe { &mut *(Rc::as_ptr(b) as *mut Variable) };
                    if let (Some(var_grad), Some(b_grad)) = (var.grad.as_ref(), &mut b_mut.grad) {
                        for i in 0..var_grad.data.len() {
                            let indices = var_grad.calc_multi_index(i);
                            let a_indices = map_indices_for_broadcast(&indices, &a.data.shape);
                            let b_indices = map_indices_for_broadcast(&indices, &b.data.shape);
                            let b_val = b.data.get_at(&b_indices);
                            b_grad.data[i] -= var_grad.data[i] * a.data.get_at(&a_indices) / (b_val * b_val);
                        }
                    }
                }
            },
            Operation::Tanh(a) => {
                if a.requires_grad {
                    let mut a_mut = unsafe { &mut *(Rc::as_ptr(a) as *mut Variable) };
                    if let (Some(var_grad), Some(a_grad)) = (var.grad.as_ref(), &mut a_mut.grad) {
                        for i in 0..var_grad.data.len() {
                            let tanh_x = var.data.data[i];
                            let dtanh = 1.0 - tanh_x * tanh_x;
                            a_grad.data[i] += var_grad.data[i] * dtanh;
                        }
                    }
                }
            },
            Operation::Relu(a) => {
                if a.requires_grad {
                    let mut a_mut = unsafe { &mut *(Rc::as_ptr(a) as *mut Variable) };
                    if let (Some(var_grad), Some(a_grad)) = (var.grad.as_ref(), &mut a_mut.grad) {
                        for i in 0..var_grad.data.len() {
                            let x = a.data.data[i];
                            let drelu = if x > 0.0 { 1.0 } else { 0.0 };
                            a_grad.data[i] += var_grad.data[i] * drelu;
                        }
                    }
                }
            }
            Operation::Sigmoid(a) => {
                if a.requires_grad {
                    let mut a_mut = unsafe { &mut *(Rc::as_ptr(a) as *mut Variable) };
                    if let (Some(var_grad), Some(a_grad)) = (var.grad.as_ref(), &mut a_mut.grad) {
                        for i in 0..var_grad.data.len() {
                            let sigmoid_x = 1.0 / (1.0 + (-var.data.data[i]).exp());
                            let dsigmoid = sigmoid_x * (1.0 - sigmoid_x);
                            a_grad.data[i] += var_grad.data[i] * dsigmoid;
                        }
                    }
                }
            },
            Operation::Pow(a, _) => {
                if a.requires_grad {
                    let mut a_mut = unsafe { &mut *(Rc::as_ptr(a) as *mut Variable) };
                    if let (Some(var_grad), Some(a_grad)) = (var.grad.as_ref(), &mut a_mut.grad) {
                        for i in 0..var_grad.data.len() {
                            let x = a.data.data[i];
                            let dx = var.data.data[i].ln() * var.data.data[i];
                            a_grad.data[i] += var_grad.data[i] * dx;
                        }
                    }
                }
            },
            Operation::None => {}
        }
    }
}

fn map_indices_for_broadcast(output_indices: &[usize], input_shape: &[usize]) -> Vec<usize> {
    let mut input_indices = vec![0; input_shape.len()];
    
    // Start from the right (least significant dimension)
    let out_offset = output_indices.len().saturating_sub(input_shape.len());
    
    for (i, &dim) in input_shape.iter().enumerate() {
        if i + out_offset < output_indices.len() {
            // If dimension is 1, use 0 index (broadcasting)
            // Otherwise use the corresponding output index
            input_indices[i] = if dim == 1 {
                0
            } else {
                output_indices[i + out_offset]
            };
        }
    }
    
    input_indices
}