# GhoseNet Tensor

`ghosenet_tensor` is a lightweight tensor computation library written in Rust. It offers a simple and extensible API for numerical computation, built with performance and learning in mind. Think of it as a minimalist NumPy-like library for Rust and is made for machine learning or deep learning with GhoseNet.

## Current Features

* **Tensor Creation**: Create tensors with custom data and shape
* **Zero Initialization**: Create zero-filled tensors using `Tensor::zeros`
* **Element Access**: Get and set elements using flattened indexing
* **Element-wise Operations**: Addition and multiplication (`add`, `mul`)
* **Shape Utilities**: Calculate flattened index with `calc_offset`
* **Well-tested**: Comes with a full test suite for core functionality
* **Autograd**: Automatic differentiation for computing gradients
* **Serialization**: Save and load tensors to/from files using `save_to_file` and `load_from_file`

## Usage

### Add to your project

```toml
[dependencies]
ghosenet_tensor = { path = "../ghosenet_tensor" } # Or from crates.io when published
```

### Create and Use Tensors

```rust
use ghosenet_tensor::Tensor;

let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let val = t.get(2);
```

### Element-wise Addition and Multiplication

```rust
use ghosenet_tensor::ops::{add, mul};

let a = Tensor::new(vec![1.0, 2.0], vec![2]);
let b = Tensor::new(vec![3.0, 4.0], vec![2]);

let result = add(a.clone(), b.clone());
let product = mul(&a, &b);
```

### Flattened Index Calculation

```rust
use ghosenet_tensor::calc_offset;

let shape = vec![2, 3];
let index = vec![1, 2];
let flat_index = calc_offset(&shape, &index); // 5
```

### Autograd

```rust
use ghosenet_tensor::{Tensor, ops::{add, mul}};

let a = Tensor::new(vec![1.0, 2.0], vec![2], true);
let b = Tensor::new(vec![3.0, 4.0], vec![2], true);

let c = add(a.clone(), b.clone());
let d = mul(&c, &a);

d.backward();
```

### Serialization

```rust
use ghosenet_tensor::Tensor;

let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

t.save_to_file("tensor.json").unwrap();

let loaded_t = Tensor::load_from_file("tensor.json").unwrap();
```

## Planned Features (coming soon)

* **Indexing**: `tensor[i]` syntax using `Index` and `IndexMut` traits
* **Operator Overloading**: `tensor1 + tensor2` using `impl Add for Tensor`
* **Shape Broadcasting**: Align shapes automatically like in NumPy
* **Multi-dimensional Indexing**: `tensor[[i, j]]` support
* **Performance Benchmarking**: Compare performance against other libraries

## Project Structure

```
src/
├── lib.rs          # Crate entry point
├── tensor.rs       # Core Tensor struct and methods
├── ops.rs          # Math operations on Tensors
├── shape.rs        # Shape utilities and helpers
├── tests.rs        # Unit tests
```

## Run Tests

```bash
cargo test
```

## Contributing

PRs, ideas, and bug reports are welcome!  
To get started with contributing to a feature above, feel free to fork and send a pull request or open a discussion.

## Author

Built with ❤️ by Ghosecorp

## License

Apache License 2.0 — free to use, modify, and distribute.