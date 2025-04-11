use crate::tensor::Tensor;

/// Broadcasts a tensor with shape [1] to match shape [n] (1D broadcasting only).
fn broadcast_to(t: &Tensor, shape: &Vec<usize>) -> Tensor {
    if t.shape == *shape {
        return t.clone();
    }
    if t.shape.len() == 1 && shape.len() == 1 && t.shape[0] == 1 {
        return Tensor::new(vec![t.data[0]; shape[0]], shape.clone());
    }
    panic!("Broadcast from {:?} to {:?} not supported", t.shape, shape);
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let shape = if a.shape == b.shape {
        a.shape.clone()
    } else if a.shape == vec![1] {
        b.shape.clone()
    } else if b.shape == vec![1] {
        a.shape.clone()
    } else {
        panic!("Broadcasting not supported for shapes {:?} and {:?}", a.shape, b.shape);
    };

    let a_b = broadcast_to(a, &shape);
    let b_b = broadcast_to(b, &shape);

    let data = a_b.data.iter().zip(&b_b.data).map(|(x, y)| x + y).collect();
    Tensor::new(data, shape)
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let shape = if a.shape == b.shape {
        a.shape.clone()
    } else if a.shape == vec![1] {
        b.shape.clone()
    } else if b.shape == vec![1] {
        a.shape.clone()
    } else {
        panic!("Broadcasting not supported for shapes {:?} and {:?}", a.shape, b.shape);
    };

    let a_b = broadcast_to(a, &shape);
    let b_b = broadcast_to(b, &shape);

    let data = a_b.data.iter().zip(&b_b.data).map(|(x, y)| x * y).collect();
    Tensor::new(data, shape)
}
