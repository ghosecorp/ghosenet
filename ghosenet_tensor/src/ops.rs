use crate::tensor::Tensor;

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape, b.shape, "Shape mismatch");
    let data = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
    Tensor::new(data, a.shape.clone())
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape, b.shape, "Shape mismatch");
    let data = a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect();
    Tensor::new(data, a.shape.clone())
}