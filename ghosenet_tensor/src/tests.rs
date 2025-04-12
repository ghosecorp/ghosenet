use crate::tensor::Tensor;
use crate::{ops::{add, mul}, calc_offset};

#[test]
fn test_tensor_creation() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let tensor = Tensor::new(data.clone(), shape.clone(), false);
    assert_eq!(tensor.data, data);
    assert_eq!(tensor.shape, shape);
}

#[test]
fn test_tensor_zeros() {
    let shape = vec![3, 2];
    let tensor = Tensor::zeros(shape.clone(), false);
    assert_eq!(tensor.data, vec![0.0; 6]);
    assert_eq!(tensor.shape, shape);
}

#[test]
fn test_tensor_get_set() {
    let mut tensor = Tensor::zeros(vec![2, 2], false);
    tensor.set(1, 42.0);
    assert_eq!(tensor.get(1), 42.0);
}

#[test]
fn test_tensor_addition() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], false);
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3], false);
    let result = add(&a, &b);
    assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_tensor_multiplication() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], false);
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3], false);
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
    let a = Tensor::new(vec![1.0, 2.0], vec![2], false);
    let b = Tensor::new(vec![3.0, 4.0, 5.0], vec![3], false);
    let _ = add(&a, &b); // should panic or error
}

#[test]
#[should_panic(expected = "Shape mismatch for broadcasting in mul")]
fn test_tensor_multiplication_shape_mismatch() {
    let a = Tensor::new(vec![1.0, 2.0], vec![2], false);
    let b = Tensor::new(vec![3.0, 4.0, 5.0], vec![3], false);
    let _ = mul(&a, &b); // should panic or error
}

#[test]
fn test_empty_tensor_creation() {
    let data: Vec<f32> = vec![];
    let shape = vec![0];
    let tensor = Tensor::new(data.clone(), shape.clone(), false);
    assert_eq!(tensor.data, data);
    assert_eq!(tensor.shape, shape);
}

#[test]
fn test_tensor_multidim_get_set() {
    let mut tensor = Tensor::zeros(vec![2, 2], false);
    let flat_index = calc_offset(&vec![2, 2], &vec![1, 1]); // Should be 3
    tensor.set(flat_index, 99.0);
    assert_eq!(tensor.get(flat_index), 99.0);
}

#[test]
fn test_calc_offset_consistency() {
    let shape = vec![2, 3];
    let tensor = Tensor::new((0..6).map(|x| x as f32).collect(), shape.clone(), false);

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let offset = calc_offset(&shape, &vec![i, j]);
            assert_eq!(tensor.data[offset], tensor.get(offset));
        }
    }
}

#[test]
fn test_tensor_clone() {
    let original = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], false);
    let cloned = original.clone();
    assert_eq!(original.data, cloned.data);
    assert_eq!(original.shape, cloned.shape);
}

#[test]
fn test_tensor_flat_indexing() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
    assert_eq!(tensor.get(0), 1.0);
    assert_eq!(tensor.get(1), 2.0);
    assert_eq!(tensor.get(3), 4.0);
}

#[test]
fn test_tensor_flat_indexing_mutation() {
    let mut tensor = Tensor::zeros(vec![2, 2], false);
    tensor.set(2, 9.0);
    assert_eq!(tensor.get(2), 9.0);
}

#[test]
fn test_tensor_multidimensional_indexing() {
    let tensor = Tensor::new((0..6).map(|x| x as f32).collect(), vec![2, 3], false);
    assert_eq!(tensor.get_at(&[0, 0]), 0.0);
    assert_eq!(tensor.get_at(&[1, 0]), 3.0);
    assert_eq!(tensor.get_at(&[1, 2]), 5.0);
}

#[test]
fn test_tensor_multidimensional_indexing_mutation() {
    let mut tensor = Tensor::zeros(vec![3, 3], false);
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
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1], false);
    let b = Tensor::new(vec![10.0, 20.0], vec![1, 2], false);

    let result = add(&a, &b);
    assert_eq!(result.shape, vec![3, 2]);
    assert_eq!(result.data, vec![11.0, 21.0, 12.0, 22.0, 13.0, 23.0]);
}

#[test]
fn test_tensor_multiplication_broadcasting() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1], false);
    let b = Tensor::new(vec![10.0, 20.0], vec![1, 2], false);

    let result = mul(&a, &b);
    assert_eq!(result.shape, vec![3, 2]);
    assert_eq!(result.data, vec![10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
}