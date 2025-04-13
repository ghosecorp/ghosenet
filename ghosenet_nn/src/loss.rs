use ghosenet_tensor::tensor::Tensor;
use ghosenet_tensor::ops::{sub, mul, div, sum, mean, log, add, select};

pub fn mse_loss(input: &Tensor, target: &Tensor) -> Tensor {
    let diff = sub(input, target);
    let squared = mul(&diff, &diff);
    mean(&squared)
}

pub fn binary_cross_entropy(input: &Tensor, target: &Tensor) -> Tensor {
    // Ensure input and target have the same shape
    assert_eq!(input.shape, target.shape, "Input and target shapes must match");
    
    // BCELoss = -1/N * sum(target * log(input) + (1 - target) * log(1 - input))
    // Clamp input to avoid log(0)
    let mut clamped_input = input.clone();
    let epsilon = 1e-7;
    for i in 0..clamped_input.data.len() {
        clamped_input.data[i] = clamped_input.data[i].max(epsilon).min(1.0 - epsilon);
    }
    
    let log_input = log(&clamped_input); // log(input)
    
    // Create a tensor filled with 1.0 with the same shape as input
    let ones = Tensor::new(vec![1.0; input.data.len()], input.shape.clone(), false);
    
    let log_complement = log(&sub(&ones, &clamped_input)); // log(1 - input)
    let term1 = mul(target, &log_input); // target * log(input)
    
    // (1 - target)
    let complement_target = sub(&ones, target);
    
    let term2 = mul(&complement_target, &log_complement); // (1 - target) * log(1 - input)
    let sum_terms = add(&term1, &term2); // term1 + term2
    
    // Calculate mean loss over the batch
    let mean_loss = mean(&sum_terms); // mean of all losses in the batch
    
    // Multiply by -1 to get the final BCE loss
    // Use a scalar with the correct shape for multiplication
    let neg_one = Tensor::new(vec![-1.0], vec![1], false);
    let neg_mean_loss = mul(&mean_loss, &neg_one); // -mean_loss
    
    neg_mean_loss
}

pub fn mae_loss(input: &Tensor, target: &Tensor) -> Tensor {
    let diff = sub(input, target);
    let abs_diff = diff.abs(); // Assuming `abs()` is implemented in `Tensor`
    mean(&abs_diff)
}

pub fn categorical_cross_entropy(pred: &Tensor, target: &Tensor) -> Tensor {
    // pred: batch_size x num_classes (probabilities)
    // target: batch_size x num_classes (one-hot encoded)
    let epsilon = 1e-7;
    let mut clamped_pred = pred.clone();
    for i in 0..clamped_pred.data.len() {
        clamped_pred.data[i] = clamped_pred.data[i].max(epsilon);
    }

    let log_pred = log(&clamped_pred);
    let loss = mul(target, &log_pred);
    let sum_loss = sum(&loss); // total loss
    let neg_mean = mul(&sum_loss, &Tensor::new(vec![-1.0], vec![], false));
    div(&neg_mean, &Tensor::new(vec![target.shape[0] as f32], vec![], false)) // mean over batch
}

pub fn huber_loss(input: &Tensor, target: &Tensor, delta: f32) -> Tensor {
    let diff = sub(input, target);
    let abs_diff = diff.abs();
    
    let condition = abs_diff.less_than_scalar(delta); // returns a mask tensor

    // 0.5 * diff^2 for small errors
    let small_error = mul(&mul(&diff, &diff), &Tensor::new(vec![0.5], vec![], false));

    // delta * (|diff| - 0.5 * delta) for large errors
    let delta_tensor = Tensor::new(vec![delta], vec![], false);
    let large_error = mul(&sub(&abs_diff, &mul(&delta_tensor, &Tensor::new(vec![0.5], vec![], false))), &delta_tensor);

    // Loss for each element in the batch
    let loss = condition.select(&small_error, &large_error); // if condition: small_error else: large_error

    loss
}


pub fn kl_divergence(p: &Tensor, q: &Tensor) -> Tensor {
    let epsilon = 1e-7;
    
    // Clamp the values of p and q to avoid log(0)
    let mut p_clamped = p.clone();
    let mut q_clamped = q.clone();
    for i in 0..p.data.len() {
        p_clamped.data[i] = p_clamped.data[i].max(epsilon);
        q_clamped.data[i] = q_clamped.data[i].max(epsilon);
    }

    // Calculate log(p / q)
    let ratio = div(&p_clamped, &q_clamped);
    let log_ratio = log(&ratio);

    // Compute p * log(p / q)
    let kl_terms = mul(&p_clamped, &log_ratio);

    // Print the intermediate terms for debugging
    println!("KL terms: {:?}", kl_terms.data);

    // Return the mean of the KL terms
    mean(&kl_terms)
}