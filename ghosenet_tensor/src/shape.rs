pub fn calc_offset(shape: &[usize], indices: &[usize]) -> usize {
    let mut offset = 0;
    let mut stride = 1;
    for (i, &dim) in shape.iter().rev().enumerate() {
        offset += indices[shape.len() - 1 - i] * stride;
        stride *= dim;
    }
    offset
}