pub mod tensor;
pub mod ops;
pub mod shape;

pub use tensor::Tensor;
pub use shape::calc_offset;

#[cfg(test)]
mod tests;