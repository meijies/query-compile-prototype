use core::CodegenContext;
use std::mem;

use arrow::{
    array::{Array, BooleanArray, Datum, Float64Array},
    buffer::{BooleanBuffer, NullBuffer},
    compute::kernels::{cmp::lt, numeric},
    datatypes::ArrowNativeTypeOp,
    error::ArrowError,
};
use cranelift::codegen::ir::{
    condcodes::{FloatCC, IntCC},
    types, AbiParam, ArgumentPurpose, InstBuilder, MemFlags,
};

pub fn hardcode_expr(a: f64, b: f64, c: f64, d: f64) -> bool {
    (a + b) / c < d
}

pub fn native_arrow_expr<T: ArrowNativeTypeOp>(a: T, b: T, c: T, d: T) -> bool {
    a.add_wrapping(b).div_wrapping(c).lt(&d)
}

pub fn jit_expr_v1() -> fn(f64, f64, f64, f64) -> bool {
    let mut ctx = CodegenContext::builder().debug().finish();
    let mut func_ctx = ctx.create_func_gen_ctx(
        "op_v1",
        vec![
            AbiParam::new(types::F64),
            AbiParam::new(types::F64),
            AbiParam::new(types::F64),
            AbiParam::new(types::F64),
        ],
        vec![AbiParam::new(types::I8)],
    );

    let entry_block = func_ctx.builder.create_block();
    func_ctx.builder.switch_to_block(entry_block);
    func_ctx
        .builder
        .append_block_params_for_function_params(entry_block);

    // call f64 add wrapping on lhs and rhs
    let lhs = func_ctx.builder.block_params(entry_block)[0];
    let rhs = func_ctx.builder.block_params(entry_block)[1];
    let sum = func_ctx.call_f64_add_wrapping(lhs, rhs);

    // call f64 div wrapping on res and f64 const.
    let rhs = func_ctx.builder.block_params(entry_block)[2];
    let div_result = func_ctx.call_f64_div_wrapping(sum, rhs);

    let to_lt = func_ctx.builder.block_params(entry_block)[3];

    let res = func_ctx.call_f64_lt(div_result, to_lt);
    let func_id = func_ctx.finalize(&[res]);
    let code = ctx.finalize(func_id);
    unsafe { mem::transmute::<_, fn(f64, f64, f64, f64) -> bool>(code) }
}

pub fn jit_expr_v2() -> fn(f64, f64, f64, f64) -> bool {
    let mut ctx = CodegenContext::builder().debug().finish();
    let mut func_ctx = ctx.create_func_gen_ctx(
        "op_v2",
        vec![
            AbiParam::new(types::F64),
            AbiParam::new(types::F64),
            AbiParam::new(types::F64),
            AbiParam::new(types::F64),
        ],
        vec![AbiParam::new(types::I8)],
    );

    let entry_block = func_ctx.builder.create_block();
    func_ctx.builder.switch_to_block(entry_block);
    func_ctx
        .builder
        .append_block_params_for_function_params(entry_block);
    // sum on lhs and rhs
    let lhs = func_ctx.builder.block_params(entry_block)[0];
    let rhs = func_ctx.builder.block_params(entry_block)[1];
    let sum = func_ctx.builder.ins().fadd(lhs, rhs);

    // div on sum and rhs
    let rhs = func_ctx.builder.block_params(entry_block)[2];
    let res = func_ctx.builder.ins().fdiv(sum, rhs);

    // lt on res and rhs
    let rhs = func_ctx.builder.block_params(entry_block)[3];
    let res = func_ctx.builder.ins().fcmp(FloatCC::LessThan, res, rhs);
    let func_id = func_ctx.finalize(&[res]);
    let code = ctx.finalize(func_id);
    unsafe { mem::transmute::<_, fn(f64, f64, f64, f64) -> bool>(code) }
}

pub fn jit_expr_on_array(
    a: &Float64Array,
    b: &Float64Array,
    c: f64,
    d: f64,
    op: fn(f64, f64, f64, f64) -> bool,
) -> Result<BooleanArray, ArrowError> {
    if a.len() != b.len() {
        return Err(ArrowError::ComputeError(
            "Cannot perform binary operation on arrays of different length".to_string(),
        ));
    }

    let nulls = NullBuffer::union(a.logical_nulls().as_ref(), b.logical_nulls().as_ref());
    let values = a
        .values()
        .iter()
        .zip(b.values())
        .map(|(l, r)| op(*l, *r, c, d));
    let buffer = BooleanBuffer::from_iter(values);
    let res = BooleanArray::new(buffer, nulls);
    Ok(res)
}
// ((a + b) / 3) > 4
pub fn native_arrow_expr_on_array(
    a: &dyn Datum,
    b: &dyn Datum,
    c: &dyn Datum,
    d: &dyn Datum,
) -> Result<BooleanArray, ArrowError> {
    let res = numeric::add(a, b).unwrap();
    let res = numeric::div(&res, c).unwrap();
    lt(&res, d)
}

pub fn jit_index_f64() -> fn(*const f64, i64) -> f64 {
    let mut ctx = CodegenContext::builder().debug().finish();
    let mut func_ctx = ctx.create_func_gen_ctx(
        "op_test",
        vec![AbiParam::new(ctx.ptype()), AbiParam::new(types::I64)],
        vec![AbiParam::new(types::F64)],
    );
    let entry_block = func_ctx.builder.create_block();
    func_ctx.builder.switch_to_block(entry_block);
    func_ctx
        .builder
        .append_block_params_for_function_params(entry_block);

    let array_ref = func_ctx.builder.block_params(entry_block)[0];
    let index = func_ctx.builder.block_params(entry_block)[1];
    let bits = func_ctx
        .builder
        .ins()
        .iconst(types::I64, types::I64.bytes() as i64);
    let index = func_ctx.builder.ins().imul(index, bits);
    let array_ref = func_ctx.builder.ins().iadd(array_ref, index);
    let value = func_ctx
        .builder
        .ins()
        .load(types::F64, MemFlags::new(), array_ref, 0);

    let func_id = func_ctx.finalize(&[value]);
    let code = ctx.finalize(func_id);

    unsafe { mem::transmute::<_, fn(*const f64, i64) -> f64>(code) }
}

pub fn jit_expr_v3() -> fn(*const u8, *const u8, *const bool, f64, f64, i64, i64) {
    let mut ctx = CodegenContext::builder().debug().finish();
    let data_type = types::F64X2;
    let result_type = types::I8X2;

    let mut func_ctx = ctx.create_func_gen_ctx(
        "op_v3",
        vec![
            AbiParam::new(ctx.ptype()),
            AbiParam::new(ctx.ptype()),
            AbiParam::special(ctx.ptype(), ArgumentPurpose::StructReturn),
            AbiParam::new(types::F64),
            AbiParam::new(types::F64),
            AbiParam::new(types::I64),
            AbiParam::new(types::I64),
        ],
        vec![],
    );
    let entry_block = func_ctx.builder.create_block();
    let body_block = func_ctx.builder.create_block();
    let exit_block = func_ctx.builder.create_block();

    func_ctx.builder.switch_to_block(entry_block);
    func_ctx
        .builder
        .append_block_params_for_function_params(entry_block);
    let p1 = func_ctx.builder.block_params(entry_block)[0];
    let p2 = func_ctx.builder.block_params(entry_block)[1];
    let p3 = func_ctx.builder.block_params(entry_block)[2];
    let p4 = func_ctx.builder.block_params(entry_block)[3];
    let p5 = func_ctx.builder.block_params(entry_block)[4];
    let p6 = func_ctx.builder.block_params(entry_block)[5];
    let p7 = func_ctx.builder.block_params(entry_block)[6];

    let p4 = func_ctx.builder.ins().splat(data_type, p4);
    let p5 = func_ctx.builder.ins().splat(data_type, p5);

    func_ctx
        .builder
        .append_block_param(body_block, func_ctx.ptype);
    func_ctx
        .builder
        .append_block_param(body_block, func_ctx.ptype);
    func_ctx
        .builder
        .append_block_param(body_block, func_ctx.ptype);
    func_ctx
        .builder
        .append_block_param(body_block, types::F64X2);
    func_ctx
        .builder
        .append_block_param(body_block, types::F64X2);
    func_ctx.builder.append_block_param(body_block, types::I64);
    func_ctx.builder.append_block_param(body_block, types::I64);
    func_ctx
        .builder
        .ins()
        .jump(body_block, &[p1, p2, p3, p4, p5, p6, p7]);
    func_ctx.builder.seal_block(entry_block);

    func_ctx.builder.switch_to_block(body_block);
    let lhs_ref = func_ctx.builder.block_params(body_block)[0];
    let rhs_ref = func_ctx.builder.block_params(body_block)[1];
    let result_ref = func_ctx.builder.block_params(body_block)[2];
    let to_div = func_ctx.builder.block_params(body_block)[3];
    let to_lt = func_ctx.builder.block_params(body_block)[4];
    let start = func_ctx.builder.block_params(body_block)[5];
    let end = func_ctx.builder.block_params(body_block)[6];

    let lhs = func_ctx
        .builder
        .ins()
        .load(data_type, MemFlags::new(), lhs_ref, 0);
    let rhs = func_ctx
        .builder
        .ins()
        .load(data_type, MemFlags::new(), rhs_ref, 0);
    let sum = func_ctx.builder.ins().fadd(lhs, rhs);
    let div_result = func_ctx.builder.ins().fdiv(sum, to_div);
    let result = func_ctx
        .builder
        .ins()
        .fcmp(FloatCC::LessThan, div_result, to_lt);

    func_ctx
        .builder
        .ins()
        .store(MemFlags::new(), result, result_ref, 0);

    let offset = func_ctx
        .builder
        .ins()
        .iconst(types::I64, data_type.bytes() as i64);

    let result_offset = func_ctx
        .builder
        .ins()
        .iconst(types::I64, result_type.bytes() as i64);

    let next_lhs_ref = func_ctx.builder.ins().iadd(offset, lhs_ref);
    let next_rhs_ref = func_ctx.builder.ins().iadd(offset, rhs_ref);
    let next_result_ref = func_ctx.builder.ins().iadd(result_offset, result_ref);

    let next_start = func_ctx.builder.ins().iadd_imm(start, 1);
    let cond = func_ctx
        .builder
        .ins()
        .icmp(IntCC::SignedLessThan, next_start, end);
    func_ctx.builder.ins().brif(
        cond,
        body_block,
        &[
            next_lhs_ref,
            next_rhs_ref,
            next_result_ref,
            to_div,
            to_lt,
            next_start,
            end,
        ],
        exit_block,
        &[],
    );
    func_ctx.builder.switch_to_block(exit_block);
    let func_id = func_ctx.finalize(&[]);
    let code = ctx.finalize(func_id);
    unsafe { mem::transmute::<_, fn(*const u8, *const u8, *const bool, f64, f64, i64, i64)>(code) }
}

pub fn jit_expr_on_array_v3(
    a: &Float64Array,
    b: &Float64Array,
    c: f64,
    d: f64,
    op: fn(*const u8, *const u8, *const bool, f64, f64, i64, i64),
) -> Result<BooleanArray, ArrowError> {
    if a.len() != b.len() {
        return Err(ArrowError::ComputeError(
            "Cannot perform binary operation on arrays of different length".to_string(),
        ));
    }

    let nulls = NullBuffer::union(a.logical_nulls().as_ref(), b.logical_nulls().as_ref());
    let a_ptr = a.values().inner().as_ptr();
    let b_ptr = b.values().inner().as_ptr();
    let mut res: Vec<bool> = Vec::with_capacity(a.len() + 1);
    let res_ptr = res.as_ptr();
    op(a_ptr, b_ptr, res_ptr, c, d, 0, (a.len() / 2) as i64);
    unsafe {
        res.set_len(a.len());
    }
    let buffer = BooleanBuffer::from_iter(res);
    Ok(BooleanArray::new(buffer, nulls))
}

pub fn hardcode_expr_on_array(
    a: &Float64Array,
    b: &Float64Array,
    c: f64,
    d: f64,
) -> Result<BooleanArray, ArrowError> {
    if a.len() != b.len() {
        return Err(ArrowError::ComputeError(
            "Cannot perform binary operation on arrays of different length".to_string(),
        ));
    }

    let nulls = NullBuffer::union(a.logical_nulls().as_ref(), b.logical_nulls().as_ref());

    let values = a
        .values()
        .iter()
        .zip(b.values())
        .map(|(l, r)| l.add_wrapping(*r).div_wrapping(c).lt(&d));
    let buffer = BooleanBuffer::from_iter(values);
    let res = BooleanArray::new(buffer, nulls);
    Ok(res)
}

#[cfg(test)]
mod tests {
    use arrow::array::Float64Array;

    use crate::expr::{jit_expr_v1, jit_expr_v2};

    use super::{jit_expr_on_array_v3, jit_expr_v3, jit_index_f64};

    #[test]
    fn test_jit_expr_v1() {
        let v1 = jit_expr_v1();
        assert!(v1(5.0_f64, 4.0_f64, 3.0_f64, 4.0_f64));
        assert!(!v1(9.0_f64, 4.0_f64, 3.0_f64, 4.0_f64));
    }

    #[test]
    fn test_jit_expr_v2() {
        let v2 = jit_expr_v2();
        assert!(v2(3.0, 4.0_f64, 3.0_f64, 4.0_f64));
        assert!(!v2(9.0_f64, 4.0_f64, 3.0_f64, 4.0_f64));
    }

    #[test]
    fn test_jit_expr_on_array_v3() {
        let a = &Float64Array::from(vec![2.0_f64, 4.0_f64]);
        let b = &Float64Array::from(vec![9.0_f64, 4.0_f64]);
        let c = 3.0_f64;
        let d = 3.0_f64;
        let op = jit_expr_v3();
        let res = jit_expr_on_array_v3(a, b, c, d, op).unwrap();
        println!("{:?}", res);
    }

    #[test]
    fn get_f64_from_array() {
        let op = jit_index_f64();
        let array = [
            1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64, 5.0_f64, 6.0_f64, 7.0_f64, 8.0_f64,
        ];
        let _ref = &array as *const f64;
        let item = op(_ref, 0_i64);
        println!("item {}", item);
        let item = op(_ref, 1_i64);
        println!("item {}", item);
        let item = op(_ref, 2_i64);
        println!("item {}", item);
    }
}
