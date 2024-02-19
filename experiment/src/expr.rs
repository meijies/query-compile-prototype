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
    condcodes::FloatCC, types, AbiParam, InstBuilder, MemFlags, StackSlotData, StackSlotKind, Type,
};

pub fn hardcode_expr(a: f64, b: f64, c: f64, d: f64) -> bool {
    (a + b) / c < d
}

pub fn native_arrow_expr<T: ArrowNativeTypeOp>(a: T, b: T, c: T, d: T) -> bool {
    a.add_wrapping(b).div_wrapping(c).lt(&d)
}

pub fn jit_expr_v1() -> fn(f64, f64, f64, f64) -> bool {
    let mut ctx = CodegenContext::default();
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
    let res = func_ctx.call_f64_add_wrapping(lhs, rhs);

    // call f64 div wrapping on res and f64 const.
    let rhs = func_ctx.builder.block_params(entry_block)[2];
    let res = func_ctx.call_f64_div_wrapping(res, rhs);

    // call f64 lt on res and f64 const.
    let slot = func_ctx
        .builder
        .func
        .create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 2));

    let res_ref = func_ctx.builder.ins().stack_addr(func_ctx.ptype, slot, 0);
    func_ctx
        .builder
        .ins()
        .store(MemFlags::new(), res, res_ref, 0);

    let rhs = func_ctx.builder.block_params(entry_block)[3];
    let rhs_ref = func_ctx.builder.ins().stack_addr(func_ctx.ptype, slot, 1);
    func_ctx
        .builder
        .ins()
        .store(MemFlags::new(), rhs, rhs_ref, 0);

    let res = func_ctx.call_f64_lt(res_ref, rhs_ref);
    let func_id = func_ctx.finalize(res);
    let code = ctx.finalize(func_id);
    unsafe { mem::transmute::<_, fn(f64, f64, f64, f64) -> bool>(code) }
}

pub fn jit_expr_v2() -> fn(f64, f64, f64, f64) -> bool {
    let mut ctx = CodegenContext::default();
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
    let func_id = func_ctx.finalize(res);
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

pub fn jit_expr_v3() -> fn(*const u8, *const u8, *const u8, *const f64, *const f64) {
    let mut ctx = CodegenContext::default();
    let data_type = types::F64.by(2).unwrap();
    let mut func_ctx = ctx.create_func_gen_ctx(
        "op_v3",
        vec![
            AbiParam::new(ctx.ptype),
            AbiParam::new(ctx.ptype),
            AbiParam::new(ctx.ptype),
            AbiParam::new(ctx.ptype),
            AbiParam::new(ctx.ptype),
        ],
        vec![],
    );

    let entry_block = func_ctx.builder.create_block();
    func_ctx.builder.switch_to_block(entry_block);
    func_ctx
        .builder
        .append_block_params_for_function_params(entry_block);

    // call f64 add wrapping on lhs and rhs
    let lhs_ref = func_ctx.builder.block_params(entry_block)[0];
    let rhs_ref = func_ctx.builder.block_params(entry_block)[1];
    let lhs = func_ctx
        .builder
        .ins()
        .load(data_type, MemFlags::new(), lhs_ref, 0);
    let rhs = func_ctx
        .builder
        .ins()
        .load(data_type, MemFlags::new(), rhs_ref, 0);
    let res = func_ctx.builder.ins().fadd(lhs, rhs);

    let div_ref = func_ctx.builder.block_params(entry_block)[3];
    let div = func_ctx
        .builder
        .ins()
        .load(data_type, MemFlags::new(), div_ref, 0);
    let res = func_ctx.builder.ins().fdiv(res, div);

    // lt on les and rh
    let rhs_ref = func_ctx.builder.block_params(entry_block)[4];
    let rhs = func_ctx
        .builder
        .ins()
        .load(data_type, MemFlags::new(), rhs_ref, 0);
    let res = func_ctx.builder.ins().fcmp(FloatCC::LessThan, res, rhs);

    let rhs = func_ctx.builder.block_params(entry_block)[2];
    func_ctx.builder.ins().store(MemFlags::new(), res, rhs, 0);
    let func_id = func_ctx.finalize_without_return();
    let code = ctx.finalize(func_id);
    unsafe {
        mem::transmute::<_, fn(*const u8, *const u8, *const u8, *const f64, *const f64)>(code)
    }
}

pub fn jit_expr_on_array_v3(
    a: &Float64Array,
    b: &Float64Array,
    // TODO use slice for moment. need to replice by shuffle ins
    c: &[f64],
    d: &[f64],
    op: fn(*const u8, *const u8, *const u8, *const f64, *const f64),
) -> Result<BooleanArray, ArrowError> {
    if a.len() != b.len() {
        return Err(ArrowError::ComputeError(
            "Cannot perform binary operation on arrays of different length".to_string(),
        ));
    }

    let nulls = NullBuffer::union(a.logical_nulls().as_ref(), b.logical_nulls().as_ref());
    let a_ptr = a.values().inner().as_ptr();
    let b_ptr = a.values().inner().as_ptr();
    let mut res: Vec<bool> = Vec::with_capacity(a.len());
    let res_ptr = res.as_ptr() as *const u8;
    op(a_ptr, b_ptr, res_ptr, c.as_ptr(), d.as_ptr());
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

    use super::{jit_expr_on_array_v3, jit_expr_v3};

    #[test]
    fn test_jit_expr_v1() {
        let v1 = jit_expr_v1();
        assert!(v1(3.0_f64, 4.0_f64, 3.0_f64, 4.0_f64));
        assert!(v1(9.0_f64, 4.0_f64, 3.0_f64, 4.0_f64));
    }

    #[test]
    fn test_jit_expr_v2() {
        let v2 = jit_expr_v2();
        assert!(v2(3.0_f64, 4.0_f64, 3.0_f64, 4.0_f64));
        assert!(!v2(9.0_f64, 4.0_f64, 3.0_f64, 4.0_f64));
    }

    #[test]
    fn test_jit_expr_on_array_v3() {
        let a = Float64Array::from(vec![3.0_f64, 4.0_f64]);
        let b = Float64Array::from(vec![3.0_f64, 4.0_f64]);
        let c = [2.0_f64, 2.0_f64];
        let d = [4.0_f64, 4.0_f64];
        let op = jit_expr_v3();
        let res = jit_expr_on_array_v3(&a, &b, &c, &d, op).unwrap();
        println!("{:?}", res);
    }
}
