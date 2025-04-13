use ghosenet_tensor::tensor::Tensor;
use std::f32;
use rand::Rng;  // Import the Rng trait

// Xavier/Glorot uniform initialization
pub fn xavier_uniform(shape: &[usize], requires_grad: bool) -> Tensor {
    let fan_in = shape[1] as f32;
    let fan_out = shape[0] as f32;
    let limit = (6.0 / (fan_in + fan_out)).sqrt();
    
    let size = shape.iter().product::<usize>();
    let mut data = Vec::with_capacity(size);
    
    let mut rng = rand::thread_rng();  // Create a random number generator

    // Use the random number generator to create values within the range [-limit, limit]
    for _ in 0..size {
        let value: f32 = rng.random_range(-limit..limit);  // Random value in the range [-limit, limit]
        data.push(value);
    }
    
    Tensor::new(data, shape.to_vec(), requires_grad)
}

// Zero initialization
pub fn zeros(shape: &[usize], requires_grad: bool) -> Tensor {
    Tensor::zeros(shape.to_vec(), requires_grad)
}

// Constant initialization
pub fn constant(shape: &[usize], value: f32, requires_grad: bool) -> Tensor {
    let size = shape.iter().product::<usize>();
    let data = vec![value; size];
    Tensor::new(data, shape.to_vec(), requires_grad)
}