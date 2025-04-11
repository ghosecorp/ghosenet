use std::ops::{Add, Mul, Index, IndexMut};
use crate::ops::{add, mul};

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product(), "Shape mismatch");
        Tensor { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Tensor {
            data: vec![0.0; size],
            shape,
        }
    }

    pub fn get(&self, index: usize) -> f32 {
        self.data[index]
    }

    pub fn set(&mut self, index: usize, value: f32) {
        self.data[index] = value;
    }

    pub fn get_at(&self, indices: &[usize]) -> f32 {
        let flat_index = self.calculate_flat_index(indices);
        self.data[flat_index]
    }

    pub fn set_at(&mut self, indices: &[usize], value: f32) {
        let flat_index = self.calculate_flat_index(indices);
        self.data[flat_index] = value;
    }

    fn calculate_flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "Dimension mismatch");
        let mut index = 0;
        let mut stride = 1;
        for (i, &dim) in self.shape.iter().rev().enumerate() {
            index += indices[self.shape.len() - 1 - i] * stride;
            stride *= dim;
        }
        index
    }

    pub fn iter(&self) -> std::slice::Iter<f32> {
        self.data.iter()
    }
}

impl Index<usize> for Tensor {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        add(&self, &rhs)
    }
}

impl Mul for Tensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        mul(&self, &rhs)
    }
}