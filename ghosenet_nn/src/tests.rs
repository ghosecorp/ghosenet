// // tests
use ghosenet_tensor::tensor::Tensor;
use crate::loss::{huber_loss, binary_cross_entropy};
use crate::layers::Linear;
use crate::activations::{ReLU, Sigmoid};
use crate::{Sequential, Module, loss::mse_loss}; // other necessary imports
// use crate::loss::huber_loss;
// use std::vec::Vec;

#[test]
fn test_simple_forward() {
    // 1. Sample input: batch of 2 samples, each with 2 features
    let input_data = vec![
        0.5, -1.2, // sample 1
        1.0,  0.8  // sample 2
    ];
    let input = Tensor::new(input_data, vec![2, 2], false);

    // 2. Define model
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 3, true)),
        Box::new(ReLU),
        Box::new(Linear::new(3, 1, true)),
        Box::new(Sigmoid),
    ]);

    // 3. Forward pass
    let output = model.forward(&input);

    // 4. Check output shape
    assert_eq!(output.shape, vec![2, 1]);

    // 5. (Optional) Compute dummy target and loss
    let target = Tensor::new(vec![0.0, 1.0], vec![2, 1], false);
    let loss = mse_loss(&output, &target);
    
    println!("Output: {:?}", output.data);
    println!("Loss: {:?}", loss.data);
}

#[test]
fn test_mse_loss_basic() {
    let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], false);
    let target = Tensor::new(vec![1.0, 2.0, 2.0], vec![3], false);
    let loss = mse_loss(&input, &target);
    // Expected: ((0)^2 + (0)^2 + (1)^2) / 3 = 1/3 = 0.333...
    assert!((loss.data[0] - 0.3333).abs() < 1e-4);
}

#[test]
fn test_binary_cross_entropy_basic() {
    let input = Tensor::new(vec![0.9, 0.2], vec![2], false);
    let target = Tensor::new(vec![1.0, 0.0], vec![2], false);
    let loss = binary_cross_entropy(&input, &target);
    // Expected: -1/2 * [log(0.9) + log(1 - 0.2)] ≈ -0.5 * [-0.105 + -0.223] = 0.164
    assert!((loss.data[0] - 0.1642).abs() < 1e-3);
}

#[cfg(test)]
mod tests {
    use super::*; // This brings everything from the current module into scope.

    #[test]
    fn test_abs_functionality() {
        let tensor = Tensor::new(vec![-2.0, 3.0, -5.0], vec![3], false);
        let abs_tensor = tensor.abs();
        assert_eq!(abs_tensor.data, vec![2.0, 3.0, 5.0]);
    }

    #[test]
    fn test_select_functionality() {
        let mask = Tensor::new(vec![1.0, 0.0], vec![2], false);
        let a = Tensor::new(vec![5.0, 6.0], vec![2], false);
        let b = Tensor::new(vec![9.0, 4.0], vec![2], false);
        let selected = mask.select(&a, &b);
        assert_eq!(selected.data, vec![5.0, 4.0]);
    }

    #[test]
    fn test_huber_loss() {
        let input = Tensor::new(vec![0.5, 1.5, 2.5], vec![3], false);
        let target = Tensor::new(vec![1.0, 1.0, 1.0], vec![3], false);
        let delta = 1.0;
        let loss = huber_loss(&input, &target, delta);
        // Assert the expected loss values (change according to your expectation)
        assert_eq!(loss.data, vec![0.125, 0.125, 1.125]); 
    }
}
