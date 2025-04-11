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