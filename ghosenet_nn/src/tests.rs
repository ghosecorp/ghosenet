// // tests
use ghosenet_tensor::tensor::Tensor;
use crate::loss::{huber_loss, binary_cross_entropy, mae_loss, kl_divergence};
// use crate::layers::{Linear, Sequential, Flatten, MaxPool2d, Dropout2d, LayerNorm, Embedding, Conv1d, Identity, BatchNorm1d};
use crate::layers::{Linear, Sequential, Flatten, Identity, Dropout};
use crate::activations::{ReLU, Sigmoid, Swish, LeakyReLU, Softmax, Tanh};
use crate::{Sequential as OtherSequential, Module, loss::mse_loss}; // other necessary imports
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
    let model = OtherSequential::new(vec![
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
fn test_relu() {
    let relu = ReLU;
    let input = Tensor::new(vec![-1.0, 0.0, 1.0, 3.5], vec![4], false);
    let output = relu.forward(&input);
    assert_eq!(output.data, vec![0.0, 0.0, 1.0, 3.5]);
}

#[test]
fn test_sigmoid() {
    let sigmoid = Sigmoid;
    let input = Tensor::new(vec![-2.0, 0.0, 2.0], vec![3], false);
    let output = sigmoid.forward(&input);
    let expected: Vec<f32> = input.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    for (o, e) in output.data.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-5);
    }
}

#[test]
fn test_tanh() {
    let tanh = Tanh;
    let input = Tensor::new(vec![-2.0, 0.0, 2.0], vec![3], false);
    let output = tanh.forward(&input);
    let expected: Vec<f32> = input.data.iter().map(|&x| x.tanh()).collect();
    for (o, e) in output.data.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-5);
    }
}

#[test]
fn test_leaky_relu() {
    let leaky_relu = LeakyReLU { negative_slope: 0.01 };
    let input = Tensor::new(vec![-2.0, 0.0, 2.0], vec![3], false);
    let output = leaky_relu.forward(&input);
    let expected = vec![-0.02, 0.0, 2.0];
    for (o, e) in output.data.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-5);
    }
}

#[test]
fn test_softmax() {
    let softmax = Softmax;
    let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], false);
    let output = softmax.forward(&input);
    let exp_vals: Vec<f32> = input.data.iter().map(|&x| x.exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let expected: Vec<f32> = exp_vals.iter().map(|&x| x / sum).collect();
    for (o, e) in output.data.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-5);
    }
}

#[test]
fn test_swish() {
    let swish = Swish;
    let input = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3], false);
    let output = swish.forward(&input);
    let expected: Vec<f32> = input.data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
    for (o, e) in output.data.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-5);
    }
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

    // Compute the binary cross-entropy loss
    let loss = binary_cross_entropy(&input, &target);

    // Compute the expected value manually
    let expected_loss = -0.5 * (
        // loss for sample 1: log(0.9) (target = 1.0)
        (1.0 * (input.data[0].ln()) + (1.0 - 1.0) * (1.0 - input.data[0]).ln()) + 
        // loss for sample 2: log(1 - 0.2) (target = 0.0)
        (0.0 * (input.data[1].ln()) + (1.0 - 0.0) * (1.0 - input.data[1]).ln())
    );

    // Compare the calculated loss to the expected loss
    assert!((loss.data[0] - expected_loss).abs() < 1e-3);
}

#[test]
fn test_mae_loss() {
    // Define input (predictions) and target tensors
    let input = Tensor::new(vec![0.5, 1.5, 2.5], vec![3], false);
    let target = Tensor::new(vec![1.0, 1.0, 1.0], vec![3], false);
    
    // Compute the MAE loss
    let loss = mae_loss(&input, &target);
    
    // Manually calculate expected loss: 
    // |0.5 - 1.0| + |1.5 - 1.0| + |2.5 - 1.0| = 0.5 + 0.5 + 1.5 = 2.5
    // Mean = 2.5 / 3 = 0.8333...
    
    assert_eq!(loss.data, vec![0.83333333]);  // Expect the mean loss value
}

#[test]
fn test_kl_divergence() {
    let p = Tensor::new(vec![0.4, 0.6], vec![2], false);  // Probabilities of p
    let q = Tensor::new(vec![0.5, 0.5], vec![2], false);  // Probabilities of q
    
    // Compute the KL divergence loss
    let loss = kl_divergence(&p, &q);
    
    // The expected value is the mean of the KL divergence terms
    // Sum of terms: 0.02013655, and the mean is 0.010068275
    assert!((loss.data[0] - 0.010068275).abs() < 1e-6, "KL Divergence Test Failed: expected 0.010068275, got {}", loss.data[0]);
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
        
        // Assert the expected loss values
        assert_eq!(loss.data, vec![0.125, 0.125, 1.0]);
    }

}

#[test]
fn test_sequential() {
    let mut seq = Sequential::new();
    seq.add(Box::new(ReLU {}));
    seq.add(Box::new(Sigmoid {}));
    let input = Tensor::new(vec![-1.0, 0.0, 2.0], vec![3], false);
    let output = seq.forward(&input);
    assert_eq!(output.data, vec![0.5, 0.5, 0.880797]);
}

#[test]
fn test_identity_layer() {
    let identity = Identity;
    let input = Tensor::new(vec![1.0, -2.5, 3.3], vec![3], false);

    // Forward should return the same data and shape
    let output = identity.forward(&input);
    assert_eq!(output.data, input.data);
    assert_eq!(output.shape, input.shape);

    // Parameters should be empty
    let params = identity.parameters();
    assert!(params.is_empty());

    // zero_grad should not panic or modify anything
    let mut identity_mut = Identity;
    identity_mut.zero_grad(); // no assertion needed; we just ensure it doesn't panic
}

#[test]
fn test_flatten_layer() {
    let flatten = Flatten;
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
    let output = flatten.forward(&input);

    assert_eq!(output.data, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(output.shape, vec![4]);
    assert_eq!(output.requires_grad, true);
    
    assert!(flatten.parameters().is_empty());
}

#[test]
fn test_dropout_eval_mode() {
    let mut dropout = Dropout::new(0.5);
    dropout.eval(); // disable dropout
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], false);
    let output = dropout.forward(&input);

    assert_eq!(output.data, input.data);
    assert_eq!(output.shape, input.shape);
}

#[test]
fn test_dropout_train_mode() {
    let mut dropout = Dropout::new(0.5);
    dropout.train(); // enable dropout
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], false);
    let output = dropout.forward(&input);

    // Check that one element is zeroed (i % 4 == 0 => 0th element)
    // and others are scaled by 1 / (1 - p) = 2.0
    assert_eq!(output.data[0], 0.0);
    assert_eq!(output.data[1], 4.0); // 2.0 * 2.0
    assert_eq!(output.data[2], 6.0); // 3.0 * 2.0
    assert_eq!(output.data[3], 8.0); // 4.0 * 2.0

    assert_eq!(output.shape, input.shape);
}

#[test]
fn test_dropout_probability_bounds() {
    let _ = Dropout::new(0.0); // should work
    let _ = Dropout::new(0.99); // should work

    // Uncommenting below should panic
    // let _ = Dropout::new(-0.1); // should panic
    // let _ = Dropout::new(1.0); // should panic
}

#[test]
fn test_dropout_no_parameters() {
    let dropout = Dropout::new(0.3);
    assert!(dropout.parameters().is_empty());
}


// #[cfg(test)]
// mod tests {
//     use ghosenet_tensor::tensor::Tensor;
//     use ghosenet_tensor::ops::*;
//     use crate::*;

//     #[test]
//     fn test_relu() {
//         let input = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2], false);
//         let relu = ReLU;
//         let output = relu.forward(&input);
//         assert_eq!(output.data, vec![0.0, 0.0, 1.0, 2.0]);
//     }

//     #[test]
//     fn test_sigmoid() {
//         let input = Tensor::new(vec![0.0], vec![1], false);
//         let sigmoid = Sigmoid;
//         let output = sigmoid.forward(&input);
//         assert!((output.data[0] - 0.5).abs() < 1e-5);
//     }

//     #[test]
//     fn test_batchnorm1d_training() {
//         let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
//         let bn = BatchNorm1d::new(2, 1e-5, 0.1);
//         let output = bn.forward(&input);
//         assert_eq!(output.shape, vec![2, 2]);
//     }

//     #[test]
//     fn test_flatten() {
//         let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
//         let flatten = Flatten;
//         let output = flatten.forward(&input);
//         assert_eq!(output.shape, vec![4]);
//     }

//     #[test]
//     fn test_identity() {
//         let input = Tensor::new(vec![1.0, 2.0], vec![2], false);
//         let id = Identity;
//         let output = id.forward(&input);
//         assert_eq!(output.data, input.data);
//     }

//     #[test]
//     fn test_conv1d() {
//         let conv = Conv1d::new(1, 1, 2, 1, 0, false);
//         let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 1, 3], false);
//         let output = conv.forward(&input);
//         assert_eq!(output.shape[2], 2);
//     }

//     #[test]
//     fn test_conv2d_shape() {
//         let conv = Conv2d::new(1, 1, (2, 2), (1, 1), (0, 0), false);
//         let input = Tensor::new(vec![1.0; 1 * 1 * 4 * 4], vec![1, 1, 4, 4], false);
//         let output = conv.forward(&input);
//         assert_eq!(output.shape, vec![1, 1, 3, 3]);
//     }

//     #[test]
//     fn test_embedding() {
//         let embedding = Embedding::new(10, 4);
//         let input = Tensor::new(vec![1.0, 3.0], vec![2], false);
//         let output = embedding.forward(&input);
//         assert_eq!(output.shape, vec![2, 4]);
//     }

//     #[test]
//     fn test_layernorm() {
//         let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3], false);
//         let norm = LayerNorm::new(3, 1e-5);
//         let output = norm.forward(&input);
//         assert_eq!(output.shape, vec![1, 3]);
//     }

//     #[test]
//     fn test_dropout2d_train_and_eval() {
//         let input = Tensor::new(vec![1.0; 2 * 3 * 4 * 4], vec![2, 3, 4, 4], false);
//         let mut dropout = Dropout2d::new(0.5);

//         dropout.train();
//         let output_train = dropout.forward(&input);
//         assert_eq!(output_train.shape, input.shape);

//         dropout.eval();
//         let output_eval = dropout.forward(&input);
//         assert_eq!(output_eval.data, input.data);
//     }

//     #[test]
//     fn test_maxpool2d() {
//         let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], false);
//         let pool = MaxPool2d::new((2, 2), None);
//         let output = pool.forward(&input);
//         assert_eq!(output.data[0], 4.0);
//         assert_eq!(output.shape, vec![1, 1, 1, 1]);
//     }

//     #[test]
//     fn test_sequential() {
//         let mut seq = Sequential::new();
//         seq.add(Box::new(ReLU));
//         seq.add(Box::new(Flatten));
//         let input = Tensor::new(vec![-1.0, 0.0, 2.0], vec![3], false);
//         let output = seq.forward(&input);
//         assert_eq!(output.data, vec![0.0, 0.0, 2.0]);
//     }
// }