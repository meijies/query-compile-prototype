use arrow::{
    array::{Array, BooleanArray, Datum, Float64Array, PrimitiveArray, Scalar}, buffer::{BooleanBuffer, NullBuffer}, compute::kernels::{cmp::lt, numeric}, datatypes::{Float64Type}, error::ArrowError, util::bench_util::create_primitive_array
};

use arrow::array::ArrowNativeTypeOp;
use criterion::*;

// ((a + b) / 3) > 4
fn arrow_expr_calc(
    a: &dyn Datum,
    b: &dyn Datum,
    c: &dyn Datum,
    d: &dyn Datum,
) -> Result<BooleanArray, ArrowError> {
    let res = numeric::add(a, b).unwrap();
    let res = numeric::div(&res, c).unwrap();
    lt(&res, d)
}

fn hardcode_expr_calc(
    a: &Float64Array,
    b: &Float64Array,
    c: &Scalar<PrimitiveArray<Float64Type>>,
    d: &Scalar<PrimitiveArray<Float64Type>>,
) -> Result<(), ArrowError> {
    if a.len() != b.len() {
        return Err(ArrowError::ComputeError(
            "Cannot perform binary operation on arrays of different length".to_string(),
        ));
    }

    if a.is_empty() {
        return Ok(());
    }

    let nulls = NullBuffer::union(a.logical_nulls().as_ref(), b.logical_nulls().as_ref());

    let values = a.values().iter().zip(b.values()).map(|(l, r)| l.add_wrapping(*r).div_wrapping(3.0).lt(&4.0));
    // JUSTIFICATION
    //  Benefit
    //      ~60% speedup
    //  Soundness
    //      `values` is an iterator with a known size from a PrimitiveArray
    let buffer = unsafe { BooleanBuffer::from_iter(values) };
    Ok(())
}

fn jit_calc(
    a: &dyn Datum,
    b: &dyn Datum,
    c: &dyn Datum,
    d: &dyn Datum,
) -> Result<BooleanArray, ArrowError> {
    todo!()
}

fn add_benchmark(c: &mut Criterion) {
    const BATCH_SIZE: usize = 64;
    let array_a = create_primitive_array::<Float64Type>(BATCH_SIZE, 0.);
    let array_b = create_primitive_array::<Float64Type>(BATCH_SIZE, 0.);
    let scalar_c = Float64Array::new_scalar(3.0);
    let scalar_d = Float64Array::new_scalar(4.0);
    c.bench_function("arrow_calc", |b| {
        b.iter(|| {
            criterion::black_box(arrow_expr_calc(&array_a, &array_b, &scalar_c, &scalar_d)).unwrap()
        })
    });

    c.bench_function("hand_calc", |b| {
        b.iter(|| {
            criterion::black_box(hardcode_expr_calc(&array_a, &array_b, &scalar_c, &scalar_d)).unwrap()
        })
    });

    // c.bench_function("jit_calc", |b| {
    //     b.iter(|| criterion::black_box(jit_calc(&array_a, &array_b, &scalar_c, &scalar_d)).unwrap())
    // });
}

criterion_group!(benches, add_benchmark);
criterion_main!(benches);
