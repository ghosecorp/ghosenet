use crate::tensor::Tensor;

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

pub fn exp(tensor: &Tensor) -> Tensor {
    let mut result = Tensor::zeros(tensor.shape.clone(), tensor.requires_grad);
    for i in 0..tensor.data.len() {
        result.data[i] = tensor.data[i].exp();
    }
    result
}

pub fn log(tensor: &Tensor) -> Tensor {
    let mut result = Tensor::zeros(tensor.shape.clone(), tensor.requires_grad);
    for i in 0..tensor.data.len() {
        result.data[i] = tensor.data[i].ln();
    }
    result
}

pub fn sum(tensor: &Tensor) -> Tensor {
    let sum_value = tensor.data.iter().sum::<f32>();
    Tensor::new(vec![sum_value], vec![], tensor.requires_grad)
}

pub fn mean(tensor: &Tensor) -> Tensor {
    let mean_value = tensor.data.iter().sum::<f32>() / tensor.data.len() as f32;
    Tensor::new(vec![mean_value], vec![], tensor.requires_grad)
}