use arrow::{
    array::Float64Array, datatypes::Float64Type, util::bench_util::create_primitive_array,
};

use criterion::*;
use experiment::expr::{
    hardcode_expr, hardcode_expr_on_array, jit_expr_on_array, jit_expr_on_array_v3, jit_expr_v1,
    jit_expr_v2, jit_expr_v3, native_arrow_expr, native_arrow_expr_on_array,
};
fn expr_bench(c: &mut Criterion) {
    let v1 = jit_expr_v1();
    let v2 = jit_expr_v2();
    c.bench_function("jit_expr_v1", |b| {
        b.iter(|| {
            v1(
                black_box(3.0_f64),
                black_box(4.0_f64),
                black_box(3.0_f64),
                black_box(4.0_f64),
            )
        })
    });
    c.bench_function("jit_expr_v2", |b| {
        b.iter(|| {
            v2(
                black_box(3.0_f64),
                black_box(4.0_f64),
                black_box(3.0_f64),
                black_box(4.0_f64),
            )
        })
    });
    c.bench_function("native_arrow_expr", |b| {
        b.iter(|| {
            native_arrow_expr::<f64>(
                black_box(3.0_f64),
                black_box(4.0_f64),
                black_box(3.0_f64),
                black_box(4.0_f64),
            )
        })
    });

    c.bench_function("hardcode_expr", |b| {
        b.iter(|| {
            hardcode_expr(
                black_box(3.0_f64),
                black_box(4.0_f64),
                black_box(3.0_f64),
                black_box(4.0_f64),
            )
        })
    });
}

fn expr_bench_on_array(c: &mut Criterion) {
    const BATCH_SIZE: usize = 64;
    let array_a = create_primitive_array::<Float64Type>(BATCH_SIZE, 0.);
    let array_b = create_primitive_array::<Float64Type>(BATCH_SIZE, 0.);
    let scalar_c = Float64Array::new_scalar(3.0);
    let scalar_d = Float64Array::new_scalar(4.0);
    let const_c = 3.0_f64;
    let const_d = 4.0_f64;
    let v1 = jit_expr_v1();
    let v2 = jit_expr_v2();
    let v3 = jit_expr_v3();

    c.bench_function("jit_esxpr_on_array_v3", |b| {
        b.iter(|| jit_expr_on_array_v3(&array_a, &array_b, 3.0f64, 4.0f64, v3).unwrap())
    });

    c.bench_function("jit_expr_on_array_v1", |b| {
        b.iter(|| {
            jit_expr_on_array(
                black_box(&array_a),
                black_box(&array_b),
                black_box(const_c),
                black_box(const_d),
                v1,
            )
            .unwrap()
        })
    });
    c.bench_function("jit_expr_on_array_v2", |b| {
        b.iter(|| {
            jit_expr_on_array(
                black_box(&array_a),
                black_box(&array_b),
                black_box(const_c),
                black_box(const_d),
                v2,
            )
            .unwrap()
        })
    });

    c.bench_function("hardcode_expr_on_array", |b| {
        b.iter(|| {
            hardcode_expr_on_array(
                black_box(&array_a),
                black_box(&array_b),
                black_box(const_c),
                black_box(const_d),
            )
            .unwrap()
        })
    });

    c.bench_function("native_arrow_expr_on_array", |b| {
        b.iter(|| {
            criterion::black_box(native_arrow_expr_on_array(
                black_box(&array_a),
                black_box(&array_b),
                black_box(&scalar_c),
                black_box(&scalar_d),
            ))
            .unwrap()
        })
    });
}

criterion_group!(
    benches,
    // expr_bench,
    expr_bench_on_array
);
criterion_main!(benches);
