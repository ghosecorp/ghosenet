use ghosenet_tensor::tensor::{Tensor, OpType};
use ghosenet_tensor::ops::{matmul, add};
use crate::{Module, Parameter};
use crate::init::{xavier_uniform, zeros};
use rand::Rng;
// use crate::activations::Activation; // Import the Activation trait

// Linear layer (fully connected)
pub struct Linear {
    pub in_features: usize,
    pub out_features: usize,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Initialize with Xavier uniform distribution
        let weight = xavier_uniform(&[out_features, in_features], true);
        
        let bias = if bias {
            // Bias shape should be [1, out_features] for broadcasting
            Some(zeros(&[1, out_features], true))
        } else {
            None
        };
        
        Linear {
            in_features,
            out_features,
            weight,
            bias,
        }
    }
    
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Ensure input has the right shape
        assert!(input.shape.len() >= 1 && input.shape.last().unwrap() == &self.in_features,
                "Expected input to have shape [..., {}], but got {:?}", 
                self.in_features, input.shape);
        
        let output = matmul(input, &self.weight.transpose());
        
        if let Some(ref bias) = self.bias {
            add(&output, bias)
        } else {
            output
        }
    }
    
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
    
    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        if let Some(ref mut bias) = self.bias {
            bias.zero_grad();
        }
    }
}

// Dropout layer
pub struct Dropout {
    pub p: f32,
    pub training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(p >= 0.0 && p < 1.0, "Dropout probability must be between 0 and 1");
        Dropout { p, training: true }
    }
    
    pub fn train(&mut self) {
        self.training = true;
    }
    
    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }
        
        // In a real implementation, we'd generate a random mask here
        // For now, we'll simulate it with a simplified approach
        let scale = 1.0 / (1.0 - self.p);
        let mut result = input.clone();
        
        // Apply dropout (this is a simplified implementation)
        // In practice, we'd use a random mask
        for i in 0..result.data.len() {
            if i % 4 != 0 { // Simple deterministic "random" for example purposes
                result.data[i] *= scale;
            } else {
                result.data[i] = 0.0;
            }
        }
        
        result
    }
    
    fn parameters(&self) -> Vec<Tensor> {
        // Dropout has no parameters
        Vec::new()
    }
    
    fn zero_grad(&mut self) {
        // Nothing to do
    }
}

// pub struct BatchNorm1d {
//     pub num_features: usize,
//     pub gamma: Tensor,
//     pub beta: Tensor,
//     pub running_mean: Tensor,
//     pub running_var: Tensor,
//     pub momentum: f32,
//     pub eps: f32,
//     pub training: bool,
// }

// impl BatchNorm1d {
//     pub fn new(num_features: usize, eps: f32, momentum: f32) -> Self {
//         BatchNorm1d {
//             num_features,
//             gamma: Tensor::ones((&[1, num_features]).to_vec(), true),
//             beta: Tensor::zeros((&[1, num_features]).to_vec(), true),
//             running_mean: Tensor::zeros((&[1, num_features]).to_vec(), false),
//             running_var: Tensor::ones((&[1, num_features]).to_vec(), false),
//             momentum,
//             eps,
//             training: true,
//         }
//     }

//     pub fn train(&mut self) {
//         self.training = true;
//     }

//     pub fn eval(&mut self) {
//         self.training = false;
//     }
// }

// impl Module for BatchNorm1d {
//     fn forward(&self, input: &Tensor) -> Tensor {
//         // Shape: [batch_size, num_features]
//         assert_eq!(input.shape[1], self.num_features, "Input features must match");
        
//         let mean = input.mean(0);
//         let var = input.var(0, self.eps);

//         let normed = (input - &mean) / (&var + self.eps).sqrt();
//         &normed * &self.gamma + &self.beta
//     }

//     fn parameters(&self) -> Vec<Tensor> {
//         vec![self.gamma.clone(), self.beta.clone()]
//     }

//     fn zero_grad(&mut self) {
//         self.gamma.zero_grad();
//         self.beta.zero_grad();
//     }
// }

// pub struct Flatten;

// impl Module for Flatten {
//     fn forward(&self, input: &Tensor) -> Tensor {
//         let new_shape = vec![input.shape.iter().product()];
//         Tensor::new(input.data.clone(), new_shape, input.requires_grad)
//     }

//     fn parameters(&self) -> Vec<Tensor> {
//         Vec::new()
//     }

//     fn zero_grad(&mut self) {}
// }

// pub struct Sequential {
//     pub layers: Vec<Box<dyn Module>>,
// }

// impl Sequential {
//     pub fn new() -> Self {
//         Sequential { layers: vec![] }
//     }

//     pub fn add(&mut self, layer: Box<dyn Module>) {
//         self.layers.push(layer);
//     }
// }

// impl Module for Sequential {
//     fn forward(&self, input: &Tensor) -> Tensor {
//         self.layers.iter().fold(input.clone(), |acc, layer| layer.forward(&acc))
//     }

//     fn parameters(&self) -> Vec<Tensor> {
//         self.layers.iter()
//             .flat_map(|layer| layer.parameters())
//             .collect()
//     }

//     fn zero_grad(&mut self) {
//         for layer in self.layers.iter_mut() {
//             layer.zero_grad();
//         }
//     }
// }

// pub struct Identity;

// impl Module for Identity {
//     fn forward(&self, input: &Tensor) -> Tensor {
//         input.clone()
//     }

//     fn parameters(&self) -> Vec<Tensor> {
//         vec![]
//     }

//     fn zero_grad(&mut self) {}
// }

// pub struct Conv1d {
//     pub in_channels: usize,
//     pub out_channels: usize,
//     pub kernel_size: usize,
//     pub stride: usize,
//     pub padding: usize,
//     pub weight: Tensor,
//     pub bias: Option<Tensor>,
// }

// impl Conv1d {
//     pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize, bias: bool) -> Self {
//         let weight = xavier_uniform(&[out_channels, in_channels, kernel_size], true);
//         let bias_tensor = if bias {
//             Some(zeros(&[out_channels], true))
//         } else {
//             None
//         };
//         Self {
//             in_channels,
//             out_channels,
//             kernel_size,
//             stride,
//             padding,
//             weight,
//             bias: bias_tensor,
//         }
//     }

//     pub fn pad_input(&self, input: &Tensor) -> Tensor {
//         // Assuming input shape [batch_size, in_channels, length]
//         let pad = self.padding;
//         if pad == 0 {
//             return input.clone();
//         }

//         let mut padded_data = vec![];
//         for b in 0..input.shape[0] {
//             for c in 0..input.shape[1] {
//                 let start = b * input.shape[1] * input.shape[2] + c * input.shape[2];
//                 let mut slice: Vec<f32> = vec![0.0; pad];
//                 slice.extend_from_slice(&input.data[start..start + input.shape[2]]);
//                 slice.extend(vec![0.0; pad]);
//                 padded_data.extend(slice);
//             }
//         }

//         Tensor::new(padded_data, vec![input.shape[0], input.shape[1], input.shape[2] + 2 * pad], input.requires_grad)
//     }
// }

// impl Module for Conv1d {
//     fn forward(&self, input: &Tensor) -> Tensor {
//         let padded = self.pad_input(input);
//         let (batch_size, _, input_len) = (padded.shape[0], padded.shape[1], padded.shape[2]);
//         let output_len = (input_len - self.kernel_size) / self.stride + 1;

//         let mut output = vec![0.0; batch_size * self.out_channels * output_len];

//         for b in 0..batch_size {
//             for o in 0..self.out_channels {
//                 for i in 0..output_len {
//                     let mut sum = 0.0;
//                     for ic in 0..self.in_channels {
//                         for k in 0..self.kernel_size {
//                             let input_idx = b * padded.shape[1] * padded.shape[2] +
//                                             ic * padded.shape[2] +
//                                             i * self.stride + k;
//                             let weight_idx = o * self.in_channels * self.kernel_size +
//                                              ic * self.kernel_size + k;
//                             sum += padded.data[input_idx] * self.weight.data[weight_idx];
//                         }
//                     }
//                     if let Some(bias) = &self.bias {
//                         sum += bias.data[o];
//                     }
//                     let out_idx = b * self.out_channels * output_len + o * output_len + i;
//                     output[out_idx] = sum;
//                 }
//             }
//         }

//         Tensor::new(output, vec![batch_size, self.out_channels, output_len], input.requires_grad)
//     }

//     fn parameters(&self) -> Vec<Tensor> {
//         let mut params = vec![self.weight.clone()];
//         if let Some(bias) = &self.bias {
//             params.push(bias.clone());
//         }
//         params
//     }

//     fn zero_grad(&mut self) {
//         self.weight.zero_grad();
//         if let Some(ref mut bias) = self.bias {
//             bias.zero_grad();
//         }
//     }
// }

// pub struct Conv2d {
//     pub in_channels: usize,
//     pub out_channels: usize,
//     pub kernel_size: (usize, usize),
//     pub stride: (usize, usize),
//     pub padding: (usize, usize),
//     pub weight: Tensor,
//     pub bias: Option<Tensor>,
// }

// impl Conv2d {
//     pub fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize), bias: bool) -> Self {
//         let weight_shape = [out_channels, in_channels, kernel_size.0, kernel_size.1];
//         let weight = xavier_uniform(&weight_shape, true);
        
//         let bias_tensor = if bias {
//             Some(zeros(&[1, out_channels, 1, 1], true))
//         } else {
//             None
//         };
        
//         Conv2d {
//             in_channels,
//             out_channels,
//             kernel_size,
//             stride,
//             padding,
//             weight,
//             bias: bias_tensor,
//         }
//     }
// }

// impl Module for Conv2d {
//     fn forward(&self, input: &Tensor) -> Tensor {
//         // Assume input shape: [batch, in_channels, height, width]
//         // You'd need to implement a proper 2D convolution operation in ops.rs
//         conv2d(input, &self.weight, self.bias.as_ref(), self.stride, self.padding)
//     }

//     fn parameters(&self) -> Vec<Tensor> {
//         let mut params = vec![self.weight.clone()];
//         if let Some(ref bias) = self.bias {
//             params.push(bias.clone());
//         }
//         params
//     }

//     fn zero_grad(&mut self) {
//         self.weight.zero_grad();
//         if let Some(ref mut bias) = self.bias {
//             bias.zero_grad();
//         }
//     }
// }

// pub struct Embedding {
//     pub num_embeddings: usize,
//     pub embedding_dim: usize,
//     pub weight: Tensor,
// }

// impl Embedding {
//     pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
//         let mut rng = rand::thread_rng();
//         let data: Vec<f32> = (0..num_embeddings * embedding_dim)
//             .map(|_| rng.gen_range(-0.1..0.1))
//             .collect();

//         Self {
//             num_embeddings,
//             embedding_dim,
//             weight: Tensor::new(data, vec![num_embeddings, embedding_dim], true),
//         }
//     }
// }

// impl Module for Embedding {
//     fn forward(&self, input: &Tensor) -> Tensor {
//         // Input is expected to be 1D or 2D with indices
//         let mut embedded = vec![];

//         for &idx in &input.data {
//             let i = idx as usize;
//             let start = i * self.embedding_dim;
//             let end = start + self.embedding_dim;
//             embedded.extend_from_slice(&self.weight.data[start..end]);
//         }

//         let out_shape = {
//             let mut s = input.shape.clone();
//             s.push(self.embedding_dim);
//             s
//         };

//         Tensor::new(embedded, out_shape, input.requires_grad)
//     }

//     fn parameters(&self) -> Vec<Tensor> {
//         vec![self.weight.clone()]
//     }

//     fn zero_grad(&mut self) {
//         self.weight.zero_grad();
//     }
// }

// pub struct LayerNorm {
//     pub normalized_shape: usize,
//     pub gamma: Tensor,
//     pub beta: Tensor,
//     pub eps: f32,
// }

// impl LayerNorm {
//     pub fn new(normalized_shape: usize, eps: f32) -> Self {
//         LayerNorm {
//             normalized_shape,
//             gamma: Tensor::ones((&[1, num_features]).to_vec(), true),
//             beta: Tensor::zeros((&[1, num_features]).to_vec(), true),
//             eps,
//         }
//     }
// }

// impl Module for LayerNorm {
//     fn forward(&self, input: &Tensor) -> Tensor {
//         let mean = input.mean(-1);
//         let var = input.var(-1, self.eps);

//         let normalized = (input - &mean) / (&var + self.eps).sqrt();
//         &normalized * &self.gamma + &self.beta
//     }

//     fn parameters(&self) -> Vec<Tensor> {
//         vec![self.gamma.clone(), self.beta.clone()]
//     }

//     fn zero_grad(&mut self) {
//         self.gamma.zero_grad();
//         self.beta.zero_grad();
//     }
// }

// pub struct Dropout2d {
//     pub p: f32,
//     pub training: bool,
// }

// impl Dropout2d {
//     pub fn new(p: f32) -> Self {
//         Dropout2d { p, training: true }
//     }

//     pub fn train(&mut self) {
//         self.training = true;
//     }

//     pub fn eval(&mut self) {
//         self.training = false;
//     }
// }

// impl Module for Dropout2d {
//     fn forward(&self, input: &Tensor) -> Tensor {
//         if !self.training || self.p == 0.0 {
//             return input.clone();
//         }

//         let mut output = input.clone();
//         let scale = 1.0 / (1.0 - self.p);

//         // This just randomly drops entire channels (dim=1)
//         let channels = input.shape[1];
//         for c in 0..channels {
//             let drop = rand::random::<f32>() < self.p;
//             if drop {
//                 for i in 0..output.data.len() {
//                     if i / input.shape[3] / input.shape[2] % channels == c {
//                         output.data[i] = 0.0;
//                     }
//                 }
//             }
//         }

//         // Scale the remaining channels
//         for val in output.data.iter_mut() {
//             *val *= scale;
//         }

//         output
//     }

//     fn parameters(&self) -> Vec<Tensor> {
//         Vec::new()
//     }

//     fn zero_grad(&mut self) {}
// }

// pub struct MaxPool2d {
//     pub kernel_size: (usize, usize),
//     pub stride: (usize, usize),
// }

// impl MaxPool2d {
//     pub fn new(kernel_size: (usize, usize), stride: Option<(usize, usize)>) -> Self {
//         MaxPool2d {
//             kernel_size,
//             stride: stride.unwrap_or(kernel_size),
//         }
//     }
// }

// impl Module for MaxPool2d {
//     fn forward(&self, input: &Tensor) -> Tensor {
//         // Shape: [batch, channels, height, width]
//         max_pool2d(input, self.kernel_size, self.stride)
//     }

//     fn parameters(&self) -> Vec<Tensor> {
//         Vec::new()
//     }

//     fn zero_grad(&mut self) {}
// }


// // pub struct RNNCell {
// //     pub input_size: usize,
// //     pub hidden_size: usize,
// //     pub weight_ih: Tensor,
// //     pub weight_hh: Tensor,
// //     pub bias_ih: Tensor,
// //     pub bias_hh: Tensor,
// //     pub activation: Box<dyn Activation>, // Store the activation function
// // }

// // impl RNNCell {
// //     pub fn new(input_size: usize, hidden_size: usize, activation: Box<dyn Activation>) -> Self {
// //         RNNCell {
// //             input_size,
// //             hidden_size,
// //             weight_ih: xavier_uniform(&[hidden_size, input_size], true),
// //             weight_hh: xavier_uniform(&[hidden_size, hidden_size], true),
// //             bias_ih: zeros(&[1, hidden_size], true),
// //             bias_hh: zeros(&[1, hidden_size], true),
// //             activation, // Pass the activation function
// //         }
// //     }

// //     pub fn step(&self, input: &Tensor, hidden: &Tensor) -> Tensor {
// //         let ih = add(&matmul(input, &self.weight_ih.transpose()), &self.bias_ih);
// //         let hh = add(&matmul(hidden, &self.weight_hh.transpose()), &self.bias_hh);
// //         self.activation.forward(&ih.add(&hh)) // Use the activation function here
// //     }
// // }

// // pub struct LSTMCell {
// //     pub input_size: usize,
// //     pub hidden_size: usize,
// //     pub weight_ih: Tensor,
// //     pub weight_hh: Tensor,
// //     pub bias_ih: Tensor,
// //     pub bias_hh: Tensor,
// //     pub activation: Box<dyn Activation>, // Configurable activation function
// // }

// // impl LSTMCell {
// //     pub fn new(input_size: usize, hidden_size: usize, activation: Box<dyn Activation>) -> Self {
// //         LSTMCell {
// //             input_size,
// //             hidden_size,
// //             weight_ih: xavier_uniform(&[hidden_size * 4, input_size], true),
// //             weight_hh: xavier_uniform(&[hidden_size * 4, hidden_size], true),
// //             bias_ih: zeros(&[1, hidden_size * 4], true),
// //             bias_hh: zeros(&[1, hidden_size * 4], true),
// //             activation,
// //         }
// //     }

// //     pub fn step(&self, input: &Tensor, hidden: &Tensor) -> Tensor {
// //         let ih = add(&matmul(input, &self.weight_ih.transpose()), &self.bias_ih);
// //         let hh = add(&matmul(hidden, &self.weight_hh.transpose()), &self.bias_hh);
// //         self.activation.forward(&ih.add(&hh)) // Use activation function
// //     }
// // }

// // pub struct GRUCell {
// //     pub input_size: usize,
// //     pub hidden_size: usize,
// //     pub weight_ih: Tensor,
// //     pub weight_hh: Tensor,
// //     pub bias_ih: Tensor,
// //     pub bias_hh: Tensor,
// //     pub activation: Box<dyn Activation>, // Configurable activation function
// // }

// // impl GRUCell {
// //     pub fn new(input_size: usize, hidden_size: usize, activation: Box<dyn Activation>) -> Self {
// //         GRUCell {
// //             input_size,
// //             hidden_size,
// //             weight_ih: xavier_uniform(&[hidden_size * 3, input_size], true),
// //             weight_hh: xavier_uniform(&[hidden_size * 3, hidden_size], true),
// //             bias_ih: zeros(&[1, hidden_size * 3], true),
// //             bias_hh: zeros(&[1, hidden_size * 3], true),
// //             activation,
// //         }
// //     }

// //     pub fn step(&self, input: &Tensor, hidden: &Tensor) -> Tensor {
// //         let ih = add(&matmul(input, &self.weight_ih.transpose()), &self.bias_ih);
// //         let hh = add(&matmul(hidden, &self.weight_hh.transpose()), &self.bias_hh);
// //         self.activation.forward(&ih.add(&hh)) // Use activation function
// //     }
// // }

