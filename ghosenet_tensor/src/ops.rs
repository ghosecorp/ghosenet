// ops.rs

use crate::tensor::{Tensor, OpType, OpNode};
use std::rc::Rc;
use std::cell::RefCell;

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let output_shape = broadcast_shapes(&a.shape, &b.shape)
        .expect("Shape mismatch for broadcasting in add");

    let mut result = Tensor::zeros(output_shape, a.requires_grad || b.requires_grad);
    
    for i in 0..result.data.len() {
        let indices = result.calc_multi_index(i);
        let a_indices = map_indices_for_broadcast(&indices, &a.shape);
        let b_indices = map_indices_for_broadcast(&indices, &b.shape);
        
        let a_value = a.get_at(&a_indices);
        let b_value = b.get_at(&b_indices);
        
        result.data[i] = a_value + b_value;
    }
    
    // Add operation tracking and backward function for autodiff
    if a.requires_grad || b.requires_grad {
        // Create shared references to inputs
        let a_rc = Rc::new(RefCell::new(a.clone()));
        let b_rc = Rc::new(RefCell::new(b.clone()));
        
        // Clone references for the closure
        let a_weak = a_rc.clone();
        let b_weak = b_rc.clone();
        let result_shape = result.shape.clone();
        
        // Create operation node
        result.op = Some(OpNode {
            op_type: OpType::Add,
            inputs: vec![a_rc, b_rc],
            backward: Some(Box::new(move |grad_output: &Tensor| {
                // Gradient of add: grad_input = grad_output
                // For each input, need to handle broadcasting
                if let Some(ref mut a_mut) = a_weak.try_borrow_mut().ok() {
                    if a_mut.requires_grad {
                        // Handle broadcasting for a's gradient
                        let mut a_grad = vec![0.0; a_mut.data.len()];
                        for i in
                            0..grad_output.data.len() {
                            let out_indices = grad_output.calc_multi_index(i);
                            let a_indices = map_indices_for_broadcast(&out_indices, &a_mut.shape);
                            let a_flat_idx = a_mut.calculate_flat_index(&a_indices);
                            a_grad[a_flat_idx] += grad_output.data[i];
                        }
                        a_mut.accumulate_grad(&a_grad);
                    }
                }
                
                if let Some(ref mut b_mut) = b_weak.try_borrow_mut().ok() {
                    if b_mut.requires_grad {
                        // Handle broadcasting for b's gradient
                        let mut b_grad = vec![0.0; b_mut.data.len()];
                        for i in 0..grad_output.data.len() {
                            let out_indices = grad_output.calc_multi_index(i);
                            let b_indices = map_indices_for_broadcast(&out_indices, &b_mut.shape);
                            let b_flat_idx = b_mut.calculate_flat_index(&b_indices);
                            b_grad[b_flat_idx] += grad_output.data[i];
                        }
                        b_mut.accumulate_grad(&b_grad);
                    }
                }
            })),
        });
        
        // Set up the gradient function that will be called during backward pass
        let result_ref = Rc::new(RefCell::new(result.clone()));
        result.grad_fn = Some(Box::new(move || {
            if let Some(ref result_tensor) = result_ref.try_borrow().ok() {
                if let Some(ref grad) = result_tensor.grad {
                    if let Some(ref op) = result_tensor.op {
                        if let Some(ref backward_fn) = op.backward {
                            // Create a temp tensor with the gradient data
                            let grad_tensor = Tensor::new(grad.clone(), result_shape.clone(), false);
                            // Call the backward function with the gradient
                            backward_fn(&grad_tensor);
                            
                            // Recursively call backward on inputs if they require grad
                            for input in &op.inputs {
                                if let Some(ref mut input_tensor) = input.try_borrow_mut().ok() {
                                    if input_tensor.requires_grad {
                                        if let Some(ref input_grad_fn) = input_tensor.grad_fn {
                                            input_grad_fn();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }));
    }
    
    result
}

pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    let output_shape = broadcast_shapes(&a.shape, &b.shape)
        .expect("Shape mismatch for broadcasting in sub");
    let mut result = Tensor::zeros(output_shape, a.requires_grad || b.requires_grad);
    
    for i in 0..result.data.len() {
        let indices = result.calc_multi_index(i);
        let a_indices = map_indices_for_broadcast(&indices, &a.shape);
        let b_indices = map_indices_for_broadcast(&indices, &b.shape);
        
        let a_value = a.get_at(&a_indices);
        let b_value = b.get_at(&b_indices);
        
        result.data[i] = a_value - b_value;
    }
    
    // Add operation tracking and backward function for autodiff
    if a.requires_grad || b.requires_grad {
        // Create shared references to inputs
        let a_rc = Rc::new(RefCell::new(a.clone()));
        let b_rc = Rc::new(RefCell::new(b.clone()));
        
        // Clone references for the closure
        let a_weak = a_rc.clone();
        let b_weak = b_rc.clone();
        let result_shape = result.shape.clone();
        
        // Create operation node
        result.op = Some(OpNode {
            op_type: OpType::Sub,
            inputs: vec![a_rc, b_rc],
            backward: Some(Box::new(move |grad_output: &Tensor| {
                // Gradient of sub: grad_a = grad_output, grad_b = -grad_output
                if let Some(ref mut a_mut) = a_weak.try_borrow_mut().ok() {
                    if a_mut.requires_grad {
                        // Handle broadcasting for a's gradient
                        let mut a_grad = vec![0.0; a_mut.data.len()];
                        for i in 0..grad_output.data.len() {
                            let out_indices = grad_output.calc_multi_index(i);
                            let a_indices = map_indices_for_broadcast(&out_indices, &a_mut.shape);
                            let a_flat_idx = a_mut.calculate_flat_index(&a_indices);
                            a_grad[a_flat_idx] += grad_output.data[i];
                        }
                        a_mut.accumulate_grad(&a_grad);
                    }
                }
                
                if let Some(ref mut b_mut) = b_weak.try_borrow_mut().ok() {
                    if b_mut.requires_grad {
                        // Handle broadcasting for b's gradient (negative)
                        let mut b_grad = vec![0.0; b_mut.data.len()];
                        for i in 0..grad_output.data.len() {
                            let out_indices = grad_output.calc_multi_index(i);
                            let b_indices = map_indices_for_broadcast(&out_indices, &b_mut.shape);
                            let b_flat_idx = b_mut.calculate_flat_index(&b_indices);
                            b_grad[b_flat_idx] -= grad_output.data[i]; // Note the negative sign
                        }
                        b_mut.accumulate_grad(&b_grad);
                    }
                }
            })),
        });
        
        // Set up the gradient function that will be called during backward pass
        let result_ref = Rc::new(RefCell::new(result.clone()));
        result.grad_fn = Some(Box::new(move || {
            if let Some(ref result_tensor) = result_ref.try_borrow().ok() {
                if let Some(ref grad) = result_tensor.grad {
                    if let Some(ref op) = result_tensor.op {
                        if let Some(ref backward_fn) = op.backward {
                            let grad_tensor = Tensor::new(grad.clone(), result_shape.clone(), false);
                            backward_fn(&grad_tensor);
                            
                            for input in &op.inputs {
                                if let Some(ref mut input_tensor) = input.try_borrow_mut().ok() {
                                    if input_tensor.requires_grad {
                                        if let Some(ref input_grad_fn) = input_tensor.grad_fn {
                                            input_grad_fn();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }));
    }
    
    result
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let output_shape = broadcast_shapes(&a.shape, &b.shape)
        .expect("Shape mismatch for broadcasting in mul");
    
    let mut result = Tensor::zeros(output_shape, a.requires_grad || b.requires_grad);
    
    for i in 0..result.data.len() {
        let indices = result.calc_multi_index(i);
        let a_indices = map_indices_for_broadcast(&indices, &a.shape);
        let b_indices = map_indices_for_broadcast(&indices, &b.shape);
        
        let a_value = a.get_at(&a_indices);
        let b_value = b.get_at(&b_indices);
        
        result.data[i] = a_value * b_value;
    }
    
    // Add operation tracking and backward function for autodiff
    if a.requires_grad || b.requires_grad {
        // Create shared references to inputs
        let a_rc = Rc::new(RefCell::new(a.clone()));
        let b_rc = Rc::new(RefCell::new(b.clone()));
        
        // Clone references for the closure
        let a_weak = a_rc.clone();
        let b_weak = b_rc.clone();
        let result_shape = result.shape.clone();
        
        // Create operation node
        result.op = Some(OpNode {
            op_type: OpType::Mul,
            inputs: vec![a_rc, b_rc],
            backward: Some(Box::new(move |grad_output: &Tensor| {
                // Gradient of mul: grad_a = grad_output * b, grad_b = grad_output * a
                if let Some(ref mut a_mut) = a_weak.try_borrow_mut().ok() {
                    if a_mut.requires_grad {
                        let b_tensor = b_weak.borrow();
                        let mut a_grad = vec![0.0; a_mut.data.len()];
                        
                        for i in 0..grad_output.data.len() {
                            let out_indices = grad_output.calc_multi_index(i);
                            let a_indices = map_indices_for_broadcast(&out_indices, &a_mut.shape);
                            let b_indices = map_indices_for_broadcast(&out_indices, &b_tensor.shape);
                            
                            let a_flat_idx = a_mut.calculate_flat_index(&a_indices);
                            let grad_val = grad_output.data[i] * b_tensor.get_at(&b_indices);
                            a_grad[a_flat_idx] += grad_val;
                        }
                        
                        a_mut.accumulate_grad(&a_grad);
                    }
                }
                
                if let Some(ref mut b_mut) = b_weak.try_borrow_mut().ok() {
                    if b_mut.requires_grad {
                        let a_tensor = a_weak.borrow();
                        let mut b_grad = vec![0.0; b_mut.data.len()];
                        
                        for i in 0..grad_output.data.len() {
                            let out_indices = grad_output.calc_multi_index(i);
                            let a_indices = map_indices_for_broadcast(&out_indices, &a_tensor.shape);
                            let b_indices = map_indices_for_broadcast(&out_indices, &b_mut.shape);
                            
                            let b_flat_idx = b_mut.calculate_flat_index(&b_indices);
                            let grad_val = grad_output.data[i] * a_tensor.get_at(&a_indices);
                            b_grad[b_flat_idx] += grad_val;
                        }
                        
                        b_mut.accumulate_grad(&b_grad);
                    }
                }
            })),
        });
        
        // Set up the gradient function that will be called during backward pass
        let result_ref = Rc::new(RefCell::new(result.clone()));
        result.grad_fn = Some(Box::new(move || {
            if let Some(ref result_tensor) = result_ref.try_borrow().ok() {
                if let Some(ref grad) = result_tensor.grad {
                    if let Some(ref op) = result_tensor.op {
                        if let Some(ref backward_fn) = op.backward {
                            let grad_tensor = Tensor::new(grad.clone(), result_shape.clone(), false);
                            backward_fn(&grad_tensor);
                            
                            for input in &op.inputs {
                                if let Some(ref mut input_tensor) = input.try_borrow_mut().ok() {
                                    if input_tensor.requires_grad {
                                        if let Some(ref input_grad_fn) = input_tensor.grad_fn {
                                            input_grad_fn();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }));
    }
    
    result
}

pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    let output_shape = broadcast_shapes(&a.shape, &b.shape)
        .expect("Shape mismatch for broadcasting in div");
    let mut result = Tensor::zeros(output_shape, a.requires_grad || b.requires_grad);
    
    for i in 0..result.data.len() {
        let indices = result.calc_multi_index(i);
        let a_indices = map_indices_for_broadcast(&indices, &a.shape);
        let b_indices = map_indices_for_broadcast(&indices, &b.shape);
        
        let a_value = a.get_at(&a_indices);
        let b_value = b.get_at(&b_indices);
        
        result.data[i] = a_value / b_value;
    }
    
    // Add operation tracking and backward function for autodiff
    if a.requires_grad || b.requires_grad {
        // Create shared references to inputs
        let a_rc = Rc::new(RefCell::new(a.clone()));
        let b_rc = Rc::new(RefCell::new(b.clone()));
        
        // Clone references for the closure
        let a_weak = a_rc.clone();
        let b_weak = b_rc.clone();
        let result_shape = result.shape.clone();
        
        // Create operation node
        result.op = Some(OpNode {
            op_type: OpType::Div,
            inputs: vec![a_rc, b_rc],
            backward: Some(Box::new(move |grad_output: &Tensor| {
                // Gradient of div: grad_a = grad_output / b, grad_b = -grad_output * a / (b^2)
                if let Some(ref mut a_mut) = a_weak.try_borrow_mut().ok() {
                    if a_mut.requires_grad {
                        let b_tensor = b_weak.borrow();
                        let mut a_grad = vec![0.0; a_mut.data.len()];
                        
                        for i in 0..grad_output.data.len() {
                            let out_indices = grad_output.calc_multi_index(i);
                            let a_indices = map_indices_for_broadcast(&out_indices, &a_mut.shape);
                            let b_indices = map_indices_for_broadcast(&out_indices, &b_tensor.shape);
                            
                            let a_flat_idx = a_mut.calculate_flat_index(&a_indices);
                            let b_value = b_tensor.get_at(&b_indices);
                            let grad_val = grad_output.data[i] / b_value;
                            a_grad[a_flat_idx] += grad_val;
                        }
                        
                        a_mut.accumulate_grad(&a_grad);
                    }
                }
                
                if let Some(ref mut b_mut) = b_weak.try_borrow_mut().ok() {
                    if b_mut.requires_grad {
                        let a_tensor = a_weak.borrow();
                        let mut b_grad = vec![0.0; b_mut.data.len()];
                        
                        for i in 0..grad_output.data.len() {
                            let out_indices = grad_output.calc_multi_index(i);
                            let a_indices = map_indices_for_broadcast(&out_indices, &a_tensor.shape);
                            let b_indices = map_indices_for_broadcast(&out_indices, &b_mut.shape);
                            
                            let b_flat_idx = b_mut.calculate_flat_index(&b_indices);
                            let a_value = a_tensor.get_at(&a_indices);
                            let b_value = b_mut.get_at(&b_indices);
                            // grad_b = -grad_output * a / b^2
                            let grad_val = -grad_output.data[i] * a_value / (b_value * b_value);
                            b_grad[b_flat_idx] += grad_val;
                        }
                        
                        b_mut.accumulate_grad(&b_grad);
                    }
                }
            })),
        });
        
        // Set up the gradient function that will be called during backward pass
        let result_ref = Rc::new(RefCell::new(result.clone()));
        result.grad_fn = Some(Box::new(move || {
            if let Some(ref result_tensor) = result_ref.try_borrow().ok() {
                if let Some(ref grad) = result_tensor.grad {
                    if let Some(ref op) = result_tensor.op {
                        if let Some(ref backward_fn) = op.backward {
                            let grad_tensor = Tensor::new(grad.clone(), result_shape.clone(), false);
                            backward_fn(&grad_tensor);
                            
                            for input in &op.inputs {
                                if let Some(ref mut input_tensor) = input.try_borrow_mut().ok() {
                                    if input_tensor.requires_grad {
                                        if let Some(ref input_grad_fn) = input_tensor.grad_fn {
                                            input_grad_fn();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }));
    }
    
    result
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape.len(), 2, "First tensor must be 2D for matmul");
    assert_eq!(b.shape.len(), 2, "Second tensor must be 2D for matmul");
    assert_eq!(a.shape[1], b.shape[0], "Inner dimensions must match for matmul");
    
    let m = a.shape[0];
    let n = b.shape[1];
    let k = a.shape[1];
    
    let mut result = Tensor::zeros(vec![m, n], a.requires_grad || b.requires_grad);
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a.get_at(&[i, p]) * b.get_at(&[p, j]);
            }
            result.set_at(&[i, j], sum);
        }
    }
    
    // Add operation tracking and backward function for autodiff
    if a.requires_grad || b.requires_grad {
        // Create shared references to inputs
        let a_rc = Rc::new(RefCell::new(a.clone()));
        let b_rc = Rc::new(RefCell::new(b.clone()));
        
        // Clone references for the closure
        let a_weak = a_rc.clone();
        let b_weak = b_rc.clone();
        let result_shape = result.shape.clone();
        
        // Create operation node
        result.op = Some(OpNode {
            op_type: OpType::MatMul,
            inputs: vec![a_rc, b_rc],
            backward: Some(Box::new(move |grad_output: &Tensor| {
                // Gradient of matmul:
                // grad_a = grad_output @ b.T
                // grad_b = a.T @ grad_output
                if let Some(ref mut a_mut) = a_weak.try_borrow_mut().ok() {
                    if a_mut.requires_grad {
                        let b_tensor = b_weak.borrow();
                        let mut a_grad = vec![0.0; a_mut.data.len()];
                        
                        // Compute grad_a = grad_output @ b.T
                        for i in 0..a_mut.shape[0] {
                            for j in 0..a_mut.shape[1] {
                                let mut sum = 0.0;
                                for k in 0..b_tensor.shape[1] {
                                    sum += grad_output.get_at(&[i, k]) * b_tensor.get_at(&[j, k]);
                                }
                                let flat_idx = a_mut.calculate_flat_index(&[i, j]);
                                a_grad[flat_idx] = sum;
                            }
                        }
                        
                        a_mut.accumulate_grad(&a_grad);
                    }
                }
                
                if let Some(ref mut b_mut) = b_weak.try_borrow_mut().ok() {
                    if b_mut.requires_grad {
                        let a_tensor = a_weak.borrow();
                        let mut b_grad = vec![0.0; b_mut.data.len()];
                        
                        // Compute grad_b = a.T @ grad_output
                        for i in 0..b_mut.shape[0] {
                            for j in 0..b_mut.shape[1] {
                                let mut sum = 0.0;
                                for k in 0..a_tensor.shape[0] {
                                    sum += a_tensor.get_at(&[k, i]) * grad_output.get_at(&[k, j]);
                                }
                                let flat_idx = b_mut.calculate_flat_index(&[i, j]);
                                b_grad[flat_idx] = sum;
                            }
                        }
                        
                        b_mut.accumulate_grad(&b_grad);
                    }
                }
            })),
        });
        
        // Set up the gradient function that will be called during backward pass
        let result_ref = Rc::new(RefCell::new(result.clone()));
        result.grad_fn = Some(Box::new(move || {
            if let Some(ref result_tensor) = result_ref.try_borrow().ok() {
                if let Some(ref grad) = result_tensor.grad {
                    if let Some(ref op) = result_tensor.op {
                        if let Some(ref backward_fn) = op.backward {
                            let grad_tensor = Tensor::new(grad.clone(), result_shape.clone(), false);
                            backward_fn(&grad_tensor);
                            
                            for input in &op.inputs {
                                if let Some(ref mut input_tensor) = input.try_borrow_mut().ok() {
                                    if input_tensor.requires_grad {
                                        if let Some(ref input_grad_fn) = input_tensor.grad_fn {
                                            input_grad_fn();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }));
    }
    
    result
}

pub fn exp(tensor: &Tensor) -> Tensor {
    let mut result = Tensor::zeros(tensor.shape.clone(), tensor.requires_grad);
    for i in 0..tensor.data.len() {
        result.data[i] = tensor.data[i].exp();
    }
    
    // Add operation tracking and backward function for autodiff
    if tensor.requires_grad {
        // Create shared reference to input
        let tensor_rc = Rc::new(RefCell::new(tensor.clone()));
        
        // Clone reference for the closure
        let tensor_weak = tensor_rc.clone();
        let result_data = result.data.clone(); // Clone result data for the closure
        let result_shape = result.shape.clone();
        
        // Create operation node
        result.op = Some(OpNode {
            op_type: OpType::Exp,
            inputs: vec![tensor_rc],
            backward: Some(Box::new(move |grad_output: &Tensor| {
                // Gradient of exp: grad_input = grad_output * exp(input)
                if let Some(ref mut tensor_mut) = tensor_weak.try_borrow_mut().ok() {
                    if tensor_mut.requires_grad {
                        let mut tensor_grad = vec![0.0; tensor_mut.data.len()];
                        
                        for i in 0..grad_output.data.len() {
                            // exp'(x) = exp(x)
                            tensor_grad[i] = grad_output.data[i] * result_data[i];
                        }
                        
                        tensor_mut.accumulate_grad(&tensor_grad);
                    }
                }
            })),
        });
        
        // Set up the gradient function
        let result_ref = Rc::new(RefCell::new(result.clone()));
        result.grad_fn = Some(Box::new(move || {
            if let Some(ref result_tensor) = result_ref.try_borrow().ok() {
                if let Some(ref grad) = result_tensor.grad {
                    if let Some(ref op) = result_tensor.op {
                        if let Some(ref backward_fn) = op.backward {
                            let grad_tensor = Tensor::new(grad.clone(), result_shape.clone(), false);
                            backward_fn(&grad_tensor);
                            
                            for input in &op.inputs {
                                if let Some(ref mut input_tensor) = input.try_borrow_mut().ok() {
                                    if input_tensor.requires_grad {
                                        if let Some(ref input_grad_fn) = input_tensor.grad_fn {
                                            input_grad_fn();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }));
    }
    
    result
}

pub fn log(tensor: &Tensor) -> Tensor {
    let mut result = Tensor::zeros(tensor.shape.clone(), tensor.requires_grad);
    for i in 0..tensor.data.len() {
        result.data[i] = tensor.data[i].ln();
    }
    
    // Add operation tracking and backward function for autodiff
    if tensor.requires_grad {
        // Create shared reference to input
        let tensor_rc = Rc::new(RefCell::new(tensor.clone()));
        
        // Clone reference for the closure
        let tensor_weak = tensor_rc.clone();
        let result_shape = result.shape.clone();
        
        // Create operation node
        result.op = Some(OpNode {
            op_type: OpType::Log,
            inputs: vec![tensor_rc],
            backward: Some(Box::new(move |grad_output: &Tensor| {
                // Gradient of log: grad_input = grad_output / input
                if let Some(ref mut tensor_mut) = tensor_weak.try_borrow_mut().ok() {
                    if tensor_mut.requires_grad {
                        let mut tensor_grad = vec![0.0; tensor_mut.data.len()];
                        
                        for i in 0..grad_output.data.len() {
                            // log'(x) = 1/x
                            tensor_grad[i] = grad_output.data[i] / tensor_mut.data[i];
                        }
                        
                        tensor_mut.accumulate_grad(&tensor_grad);
                    }
                }
            })),
        });
        
        // Set up the gradient function
        let result_ref = Rc::new(RefCell::new(result.clone()));
        result.grad_fn = Some(Box::new(move || {
            if let Some(ref result_tensor) = result_ref.try_borrow().ok() {
                if let Some(ref grad) = result_tensor.grad {
                    if let Some(ref op) = result_tensor.op {
                        if let Some(ref backward_fn) = op.backward {
                            let grad_tensor = Tensor::new(grad.clone(), result_shape.clone(), false);
                            backward_fn(&grad_tensor);
                            
                            for input in &op.inputs {
                                if let Some(ref mut input_tensor) = input.try_borrow_mut().ok() {
                                    if input_tensor.requires_grad {
                                        if let Some(ref input_grad_fn) = input_tensor.grad_fn {
                                            input_grad_fn();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }));
    }
    
    result
}

pub fn sum(tensor: &Tensor) -> Tensor {
    let sum_value = tensor.data.iter().sum::<f32>();
    let mut result = Tensor::new(vec![sum_value], vec![], tensor.requires_grad);
    
    // Add operation tracking and backward function for autodiff
    if tensor.requires_grad {
        // Create shared reference to input
        let tensor_rc = Rc::new(RefCell::new(tensor.clone()));
        
        // Clone reference for the closure
        let tensor_weak = tensor_rc.clone();
        let result_shape = result.shape.clone();
        
        // Create operation node
        result.op = Some(OpNode {
            op_type: OpType::Sum,
            inputs: vec![tensor_rc],
            backward: Some(Box::new(move |grad_output: &Tensor| {
                // Gradient of sum: grad_input = grad_output broadcasted to input shape
                if let Some(ref mut tensor_mut) = tensor_weak.try_borrow_mut().ok() {
                    if tensor_mut.requires_grad {
                        // For sum, the gradient is the same value repeated for each input element
                        let grad_value = grad_output.data[0]; // Sum produces a scalar
                        let tensor_grad = vec![grad_value; tensor_mut.data.len()];
                        
                        tensor_mut.accumulate_grad(&tensor_grad);
                    }
                }
            })),
        });
        
        // Set up the gradient function
        let result_ref = Rc::new(RefCell::new(result.clone()));
        result.grad_fn = Some(Box::new(move || {
            if let Some(ref result_tensor) = result_ref.try_borrow().ok() {
                if let Some(ref grad) = result_tensor.grad {
                    if let Some(ref op) = result_tensor.op {
                        if let Some(ref backward_fn) = op.backward {
                            let grad_tensor = Tensor::new(grad.clone(), result_shape.clone(), false);
                            backward_fn(&grad_tensor);
                            
                            for input in &op.inputs {
                                if let Some(ref mut input_tensor) = input.try_borrow_mut().ok() {
                                    if input_tensor.requires_grad {
                                        if let Some(ref input_grad_fn) = input_tensor.grad_fn {
                                            input_grad_fn();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }));
    }
    
    result
}

pub fn mean(tensor: &Tensor) -> Tensor {
    let mean_value = tensor.data.iter().sum::<f32>() / tensor.data.len() as f32;
    let mut result = Tensor::new(vec![mean_value], vec![], tensor.requires_grad);
    
    // Add operation tracking and backward function for autodiff
    if tensor.requires_grad {
        // Create shared reference to input
        let tensor_rc = Rc::new(RefCell::new(tensor.clone()));
        
        // Clone reference for the closure
        let tensor_weak = tensor_rc.clone();
        let result_shape = result.shape.clone();
        let tensor_len = tensor.data.len() as f32;
        
        // Create operation node
        result.op = Some(OpNode {
            op_type: OpType::Mean,
            inputs: vec![tensor_rc],
            backward: Some(Box::new(move |grad_output: &Tensor| {
                // Gradient of mean: grad_input = grad_output / n (broadcasted)
                if let Some(ref mut tensor_mut) = tensor_weak.try_borrow_mut().ok() {
                    if tensor_mut.requires_grad {
                        // For mean, the gradient is (grad_output / n) for each input element
                        let grad_value = grad_output.data[0] / tensor_len; // Mean produces a scalar
                        let tensor_grad = vec![grad_value; tensor_mut.data.len()];
                        
                        tensor_mut.accumulate_grad(&tensor_grad);
                    }
                }
            })),
        });
        
        // Set up the gradient function
        let result_ref = Rc::new(RefCell::new(result.clone()));
        result.grad_fn = Some(Box::new(move || {
            if let Some(ref result_tensor) = result_ref.try_borrow().ok() {
                if let Some(ref grad) = result_tensor.grad {
                    if let Some(ref op) = result_tensor.op {
                        if let Some(ref backward_fn) = op.backward {
                            let grad_tensor = Tensor::new(grad.clone(), result_shape.clone(), false);
                            backward_fn(&grad_tensor);
                            
                            for input in &op.inputs {
                                if let Some(ref mut input_tensor) = input.try_borrow_mut().ok() {
                                    if input_tensor.requires_grad {
                                        if let Some(ref input_grad_fn) = input_tensor.grad_fn {
                                            input_grad_fn();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }));
    }
    
    result
}

// Helper function to map output indices to input indices for broadcasted dimensions
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

pub fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
    let mut result = vec![];
    let max_len = std::cmp::max(shape1.len(), shape2.len());
    
    for i in 0..max_len {
        let dim1 = *shape1.get(shape1.len().saturating_sub(i + 1)).unwrap_or(&1);
        let dim2 = *shape2.get(shape2.len().saturating_sub(i + 1)).unwrap_or(&1);
        
        if dim1 == dim2 || dim1 == 1 || dim2 == 1 {
            result.push(std::cmp::max(dim1, dim2));
        } else {
            return None; // incompatible shapes
        }
    }
    
    result.reverse();
    Some(result)
}

pub fn select(mask: &Tensor, if_true: &Tensor, if_false: &Tensor) -> Tensor {
    assert_eq!(mask.shape, if_true.shape);
    assert_eq!(if_true.shape, if_false.shape);

    let selected_data: Vec<f32> = mask
        .data
        .iter()
        .zip(&if_true.data)
        .zip(&if_false.data)
        .map(|((&m, &t), &f)| if m == 1.0 { t } else { f })
        .collect();

    let mut result = Tensor::new(
        selected_data,
        mask.shape.clone(),
        if_true.requires_grad || if_false.requires_grad,
    );

    if result.requires_grad {
        let mask_rc = Rc::new(RefCell::new(mask.clone()));
        let true_rc = Rc::new(RefCell::new(if_true.clone()));
        let false_rc = Rc::new(RefCell::new(if_false.clone()));

        let mask_clone = Rc::clone(&mask_rc);
        let true_clone = Rc::clone(&true_rc);
        let false_clone = Rc::clone(&false_rc);

        result.op = Some(OpNode {
            op_type: OpType::Input, // Or a new variant like `Select` if you want
            inputs: vec![Rc::clone(&mask_rc), Rc::clone(&true_rc), Rc::clone(&false_rc)],
            backward: Some(Box::new(move |grad: &Tensor| {
                let mask_borrow = mask_clone.borrow();
                let true_borrow = true_clone.borrow();
                let false_borrow = false_clone.borrow();

                let mut grad_true = vec![0.0; grad.data.len()];
                let mut grad_false = vec![0.0; grad.data.len()];

                for i in 0..grad.data.len() {
                    if mask_borrow.data[i] == 1.0 {
                        grad_true[i] = grad.data[i];
                    } else {
                        grad_false[i] = grad.data[i];
                    }
                }

                let t_grad = Tensor::new(grad_true, grad.shape.clone(), false);
                let f_grad = Tensor::new(grad_false, grad.shape.clone(), false);

                // Accumulate gradients
                if true_borrow.requires_grad {
                    let mut true_mut = true_clone.borrow_mut();
                    if let Some(ref mut g) = true_mut.grad {
                        for i in 0..g.len() {
                            g[i] += t_grad.data[i];
                        }
                    } else {
                        true_mut.grad = Some(t_grad.data.clone());
                    }
                }

                if false_borrow.requires_grad {
                    let mut false_mut = false_clone.borrow_mut();
                    if let Some(ref mut g) = false_mut.grad {
                        for i in 0..g.len() {
                            g[i] += f_grad.data[i];
                        }
                    } else {
                        false_mut.grad = Some(f_grad.data.clone());
                    }
                }
            })),
        });
    }

    result
}