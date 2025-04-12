use crate::tensor::Tensor;
use crate::{ops::{add, mul}, calc_offset};

#[test]
fn test_tensor_creation() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let tensor = Tensor::new(data.clone(), shape.clone());
    assert_eq!(tensor.data, data);
    assert_eq!(tensor.shape, shape);
}

#[test]
fn test_tensor_zeros() {
    let shape = vec![3, 2];
    let tensor = Tensor::zeros(shape.clone());
    assert_eq!(tensor.data, vec![0.0; 6]);
    assert_eq!(tensor.shape, shape);
}

#[test]
fn test_tensor_get_set() {
    let mut tensor = Tensor::zeros(vec![2, 2]);
    tensor.set(1, 42.0);
    assert_eq!(tensor.get(1), 42.0);
}

#[test]
fn test_tensor_addition() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
    let result = add(&a, &b);
    assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_tensor_multiplication() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
    let result = mul(&a, &b);
    assert_eq!(result.data, vec![4.0, 10.0, 18.0]);
}

#[test]
fn test_calc_offset() {
    let shape = vec![3, 4, 5]; // Shape: 3x4x5
    let indices = vec![1, 2, 3];
    let offset = calc_offset(&shape, &indices);
    // offset = 1*(4*5) + 2*(5) + 3 = 20 + 10 + 3 = 33
    assert_eq!(offset, 33);
}

#[test]
#[should_panic(expected = "Shape mismatch for broadcasting in add")] // adapt this if you're using Result instead of panic
fn test_tensor_addition_shape_mismatch() {
    let a = Tensor::new(vec![1.0, 2.0], vec![2]);
    let b = Tensor::new(vec![3.0, 4.0, 5.0], vec![3]);
    let _ = add(&a, &b); // should panic or error
}

#[test]
#[should_panic(expected = "Shape mismatch for broadcasting in mul")]
fn test_tensor_multiplication_shape_mismatch() {
    let a = Tensor::new(vec![1.0, 2.0], vec![2]);
    let b = Tensor::new(vec![3.0, 4.0, 5.0], vec![3]);
    let _ = mul(&a, &b); // should panic or error
}

#[test]
fn test_empty_tensor_creation() {
    let data: Vec<f32> = vec![];
    let shape = vec![0];
    let tensor = Tensor::new(data.clone(), shape.clone());
    assert_eq!(tensor.data, data);
    assert_eq!(tensor.shape, shape);
}

#[test]
fn test_tensor_multidim_get_set() {
    let mut tensor = Tensor::zeros(vec![2, 2]);
    let flat_index = calc_offset(&vec![2, 2], &vec![1, 1]); // Should be 3
    tensor.set(flat_index, 99.0);
    assert_eq!(tensor.get(flat_index), 99.0);
}

#[test]
fn test_calc_offset_consistency() {
    let shape = vec![2, 3];
    let tensor = Tensor::new((0..6).map(|x| x as f32).collect(), shape.clone());

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let offset = calc_offset(&shape, &vec![i, j]);
            assert_eq!(tensor.data[offset], tensor.get(offset));
        }
    }
}

#[test]
fn test_tensor_clone() {
    let original = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let cloned = original.clone();
    assert_eq!(original.data, cloned.data);
    assert_eq!(original.shape, cloned.shape);
}

#[test]
fn test_tensor_flat_indexing() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(tensor.get(0), 1.0);
    assert_eq!(tensor.get(1), 2.0);
    assert_eq!(tensor.get(3), 4.0);
}

#[test]
fn test_tensor_flat_indexing_mutation() {
    let mut tensor = Tensor::zeros(vec![2, 2]);
    tensor.set(2, 9.0);
    assert_eq!(tensor.get(2), 9.0);
}

#[test]
fn test_tensor_multidimensional_indexing() {
    let tensor = Tensor::new((0..6).map(|x| x as f32).collect(), vec![2, 3]);
    assert_eq!(tensor.get_at(&[0, 0]), 0.0);
    assert_eq!(tensor.get_at(&[1, 0]), 3.0);
    assert_eq!(tensor.get_at(&[1, 2]), 5.0);
}

#[test]
fn test_tensor_multidimensional_indexing_mutation() {
    let mut tensor = Tensor::zeros(vec![3, 3]);
    tensor.set_at(&[2, 1], 7.0);
    assert_eq!(tensor.get_at(&[2, 1]), 7.0);
}

// #[test]
// #[should_panic(expected = "Index out of bounds")]
// fn test_tensor_flat_index_out_of_bounds() {
//     let tensor = Tensor::zeros(vec![2, 2]);
//     tensor.get(5); // Invalid
// }

// #[test]
// #[should_panic(expected = "Index out of bounds")]
// fn test_tensor_multidimensional_index_out_of_bounds() {
//     let tensor = Tensor::zeros(vec![2, 2]);
//     tensor.get_at(&[2, 1]); // Invalid
// }


#[test]
fn test_tensor_addition_broadcasting() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
    let b = Tensor::new(vec![10.0, 20.0], vec![1, 2]);

    let result = add(&a, &b);
    assert_eq!(result.shape, vec![3, 2]);
    assert_eq!(result.data, vec![11.0, 21.0, 12.0, 22.0, 13.0, 23.0]);
}

#[test]
fn test_tensor_multiplication_broadcasting() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
    let b = Tensor::new(vec![10.0, 20.0], vec![1, 2]);

    let result = mul(&a, &b);
    assert_eq!(result.shape, vec![3, 2]);
    assert_eq!(result.data, vec![10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
}

// #[cfg(test)]
// mod autograd_tests {
//     use super::*;
//     use crate::tensor::Tensor;
//     use crate::autograd::{Variable, backward};
//     use std::rc::Rc;

//     #[test]
//     fn test_scalar_addition_backward() {
//         // Create variables
//         let a = Rc::new(Variable::from_scalar(2.0).requires_grad());
//         let b = Rc::new(Variable::from_scalar(3.0).requires_grad());
        
//         // Forward pass
//         let c = a.clone() + b.clone();
        
//         // Backward pass
//         backward(c.clone(), None);
        
//         // Check gradients
//         assert_eq!(a.grad.as_ref().unwrap().data[0], 1.0);
//         assert_eq!(b.grad.as_ref().unwrap().data[0], 1.0);
//     }

//     #[test]
//     fn test_scalar_multiplication_backward() {
//         // Create variables
//         let a = Rc::new(Variable::from_scalar(2.0).requires_grad());
//         let b = Rc::new(Variable::from_scalar(3.0).requires_grad());
        
//         // Forward pass
//         let c = a.clone() * b.clone();
        
//         // Backward pass
//         backward(c.clone(), None);
        
//         // Check gradients
//         assert_eq!(a.grad.as_ref().unwrap().data[0], 3.0); // dc/da = b = 3
//         assert_eq!(b.grad.as_ref().unwrap().data[0], 2.0); // dc/db = a = 2
//     }

//     #[test]
//     fn test_scalar_chain_rule() {
//         // Create variables
//         let a = Rc::new(Variable::from_scalar(2.0).requires_grad());
//         let b = Rc::new(Variable::from_scalar(3.0).requires_grad());
//         let c = Rc::new(Variable::from_scalar(4.0).requires_grad());
        
//         // Forward pass: f = a * b + c
//         let temp = a.clone() * b.clone();
//         let f = temp + c.clone();
        
//         // Backward pass
//         backward(f.clone(), None);
        
//         // Check gradients
//         assert_eq!(a.grad.as_ref().unwrap().data[0], 3.0); // df/da = b = 3
//         assert_eq!(b.grad.as_ref().unwrap().data[0], 2.0); // df/db = a = 2
//         assert_eq!(c.grad.as_ref().unwrap().data[0], 1.0); // df/dc = 1
//     }

//     #[test]
//     fn test_vector_addition_backward() {
//         // Create vector variables
//         let a_tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
//         let b_tensor = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        
//         let a = Rc::new(Variable::new(a_tensor).requires_grad());
//         let b = Rc::new(Variable::new(b_tensor).requires_grad());
        
//         // Forward pass
//         let c = a.clone() + b.clone();
        
//         // Create gradient for output
//         let grad_output = Some(Tensor::new(vec![1.0, 1.0, 1.0], vec![3]));
        
//         // Backward pass
//         backward(c.clone(), grad_output);
        
//         // Check gradients
//         let a_grad = a.grad.as_ref().unwrap();
//         let b_grad = b.grad.as_ref().unwrap();
        
//         assert_eq!(a_grad.data, vec![1.0, 1.0, 1.0]);
//         assert_eq!(b_grad.data, vec![1.0, 1.0, 1.0]);
//     }

//     #[test]
//     fn test_vector_multiplication_backward() {
//         // Create vector variables
//         let a_tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
//         let b_tensor = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        
//         let a = Rc::new(Variable::new(a_tensor).requires_grad());
//         let b = Rc::new(Variable::new(b_tensor).requires_grad());
        
//         // Forward pass
//         let c = a.clone() * b.clone();
        
//         // Create gradient for output
//         let grad_output = Some(Tensor::new(vec![1.0, 1.0, 1.0], vec![3]));
        
//         // Backward pass
//         backward(c.clone(), grad_output);
        
//         // Check gradients
//         let a_grad = a.grad.as_ref().unwrap();
//         let b_grad = b.grad.as_ref().unwrap();
        
//         // For each element i: da_i = db_i * grad_i
//         assert_eq!(a_grad.data, vec![4.0, 5.0, 6.0]);
//         // For each element i: db_i = da_i * grad_i
//         assert_eq!(b_grad.data, vec![1.0, 2.0, 3.0]);
//     }

    // #[test]
    // fn test_broadcasting_backward() {
    //     // Create variables with shapes that require broadcasting
    //     let a_tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
    //     let b_tensor = Tensor::new(vec![4.0, 5.0], vec![1, 2]);
        
    //     let a = Rc::new(Variable::new(a_tensor).requires_grad());
    //     let b = Rc::new(Variable::new(b_tensor).requires_grad());
        
    //     // Forward pass
    //     let c = a.clone() + b.clone();
        
    //     // Create gradient for output (3x2 tensor with all 1s)
    //     let grad_data = vec![1.0; 6];
    //     let grad_output = Some(Tensor::new(grad_data, vec![3, 2]));
        
    //     // Backward pass
    //     backward(c.clone(), grad_output);
        
    //     // Check gradients
    //     let a_grad = a.grad.as_ref().unwrap();
    //     let b_grad = b.grad.as_ref().unwrap();
        
    //     // Each element in a is added to 2 elements in the output, so gets gradient 2
    //     assert_eq!(a_grad.data, vec![2.0, 2.0, 2.0]);
        
    //     // Each element in b is added to 3 elements in the output, so gets gradient 3
    //     assert_eq!(b_grad.data, vec![3.0, 3.0]);
    // }

    // #[test]
    // fn test_zero_grad() {
    //     // Create variable
    //     let mut var = Variable::from_scalar(2.0).requires_grad();
        
    //     // Set some gradient
    //     if let Some(grad) = &mut var.grad {
    //         grad.data[0] = 5.0;
    //     }
        
    //     // Verify gradient exists
    //     assert_eq!(var.grad.as_ref().unwrap().data[0], 5.0);
        
    //     // Zero out gradient
    //     var.zero_grad();
        
    //     // Verify gradient is zeroed
    //     assert_eq!(var.grad.as_ref().unwrap().data[0], 0.0);
    // }

    // #[test]
    // fn test_gradient_step() {
    //     // Create variable
    //     let mut var = Variable::from_scalar(10.0).requires_grad();
        
    //     // Set some gradient
    //     if let Some(grad) = &mut var.grad {
    //         grad.data[0] = 2.0;
    //     }
        
    //     // Apply gradient step with learning rate 0.1
    //     var.step(0.1);
        
    //     // Verify parameter update: 10.0 - 0.1 * 2.0 = 9.8
    //     assert_eq!(var.data.data[0], 9.8);
    // }

    // #[test]
    // fn test_simple_neural_network() {
    //     // Create parameters
    //     let w1 = Rc::new(Variable::from_scalar(2.0).requires_grad());
    //     let w2 = Rc::new(Variable::from_scalar(3.0).requires_grad());
    //     let b = Rc::new(Variable::from_scalar(1.0).requires_grad());
        
    //     // Input
    //     let x1 = Rc::new(Variable::from_scalar(1.0));
    //     let x2 = Rc::new(Variable::from_scalar(2.0));
        
    //     // Forward pass (linear model): y = w1*x1 + w2*x2 + b
    //     let wx1 = w1.clone() * x1.clone();
    //     let wx2 = w2.clone() * x2.clone();
    //     let y_pred = wx1 + wx2 + b.clone();
        
    //     // Target
    //     let y_target = 10.0;
        
    //     // Loss: L = (y_pred - y_target)^2
    //     let loss = Rc::new(Variable::from_scalar((y_pred.data.data[0] - y_target).powi(2)).requires_grad());
        
    //     // Backward pass
    //     backward(loss.clone(), None);
        
    //     // Check gradients
    //     // dL/dw1 = 2 * (y_pred - y_target) * x1
    //     // dL/dw2 = 2 * (y_pred - y_target) * x2
    //     // dL/db = 2 * (y_pred - y_target)
        
    //     let prediction_error = y_pred.data.data[0] - y_target; // Should be 2*1 + 3*2 + 1 - 10 = -1
    //     assert!((prediction_error - (-1.0)).abs() < 1e-6);
        
    //     let expected_w1_grad = 2.0 * prediction_error * x1.data.data[0]; // 2 * (-1) * 1 = -2
    //     let expected_w2_grad = 2.0 * prediction_error * x2.data.data[0]; // 2 * (-1) * 2 = -4
    //     let expected_b_grad = 2.0 * prediction_error; // 2 * (-1) = -2
        
    //     assert!((w1.grad.as_ref().unwrap().data[0] - expected_w1_grad).abs() < 1e-6);
    //     assert!((w2.grad.as_ref().unwrap().data[0] - expected_w2_grad).abs() < 1e-6);
    //     assert!((b.grad.as_ref().unwrap().data[0] - expected_b_grad).abs() < 1e-6);
    // }
// }