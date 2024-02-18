use arrow::{
    array::Float64Array, datatypes::Float64Type, util::bench_util::create_primitive_array,
};

use criterion::*;
use experiment::expr::{arrow_expr_calc, create_jit_op, hardcode_expr_calc, jit_expr_calc};

fn add_benchmark(c: &mut Criterion) {
    const BATCH_SIZE: usize = 64;
    let array_a = create_primitive_array::<Float64Type>(BATCH_SIZE, 0.);
    let array_b = create_primitive_array::<Float64Type>(BATCH_SIZE, 0.);
    let scalar_c = Float64Array::new_scalar(3.0);
    let scalar_d = Float64Array::new_scalar(4.0);
    let op = create_jit_op();
    c.bench_function("arrow_calc", |b| {
        b.iter(|| {
            criterion::black_box(arrow_expr_calc(&array_a, &array_b, &scalar_c, &scalar_d)).unwrap()
        })
    });

    c.bench_function("hand_calc", |b| {
        b.iter(|| criterion::black_box(hardcode_expr_calc(&array_a, &array_b)).unwrap())
    });

    c.bench_function("jit_calc", |b| {
        b.iter(|| criterion::black_box(jit_expr_calc(&array_a, &array_b, op)).unwrap())
    });
}

criterion_group!(benches, add_benchmark);
criterion_main!(benches);
