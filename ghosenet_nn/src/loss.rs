use ghosenet_tensor::tensor::Tensor;
use ghosenet_tensor::ops::{sub, mul, div, sum, mean, log, add, select};

pub fn mse_loss(input: &Tensor, target: &Tensor) -> Tensor {
    let diff = sub(input, target);
    let squared = mul(&diff, &diff);
    mean(&squared)
}

pub fn binary_cross_entropy(input: &Tensor, target: &Tensor) -> Tensor {
    // BCELoss = -1/N * sum(target * log(input) + (1 - target) * log(1 - input))
    // Clamp input to avoid log(0)
    let mut clamped_input = input.clone();
    let epsilon = 1e-7;
    for i in 0..clamped_input.data.len() {
        clamped_input.data[i] = clamped_input.data[i].max(epsilon).min(1.0 - epsilon);
    }
    
    let log_input = log(&clamped_input);
    let log_complement = log(&sub(&Tensor::new(vec![1.0], vec![], false), &clamped_input));
    
    let term1 = mul(target, &log_input);
    let complement_target = sub(&Tensor::new(vec![1.0], target.shape.clone(), false), target);
    let term2 = mul(&complement_target, &log_complement);
    
    let sum_terms = add(&term1, &term2);
    let neg_mean = mul(&mean(&sum_terms), &Tensor::new(vec![-1.0], vec![], false));
    
    neg_mean
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

    let loss = condition.select(&small_error, &large_error); // if condition: small_error else: large_error
    mean(&loss)
}

pub fn kl_divergence(p: &Tensor, q: &Tensor) -> Tensor {
    let epsilon = 1e-7;
    let mut p_clamped = p.clone();
    let mut q_clamped = q.clone();
    for i in 0..p.data.len() {
        p_clamped.data[i] = p_clamped.data[i].max(epsilon);
        q_clamped.data[i] = q_clamped.data[i].max(epsilon);
    }

    let log_ratio = log(&div(&p_clamped, &q_clamped));
    let kl = mul(&p_clamped, &log_ratio);
    mean(&kl)
}
