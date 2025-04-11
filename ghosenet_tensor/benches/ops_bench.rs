use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ghosenet_tensor::tensor::Tensor;
use ghosenet_tensor::ops::{add, mul};

fn add_benchmark(c: &mut Criterion) {
    let size = 100_000;
    let a = Tensor::new(vec![1.0; size], vec![size]);
    let b = Tensor::new(vec![2.0; size], vec![size]);

    c.bench_function("tensor add", |bench| {
        bench.iter(|| {
            let _ = add(black_box(&a), black_box(&b));
        });
    });
}

fn mul_benchmark(c: &mut Criterion) {
    let size = 100_000;
    let a = Tensor::new(vec![1.0; size], vec![size]);
    let b = Tensor::new(vec![2.0; size], vec![size]);

    c.bench_function("tensor mul", |bench| {
        bench.iter(|| {
            let _ = mul(black_box(&a), black_box(&b));
        });
    });
}

criterion_group!(tensor_ops, add_benchmark, mul_benchmark);
criterion_main!(tensor_ops);