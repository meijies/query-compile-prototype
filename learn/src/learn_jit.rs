use arrow::{
    array::{Array, ArrayRef, AsArray, Datum, Int32Array},
    compute::kernels::numeric,
    error::ArrowError,
};
use cranelift::{
    codegen::ir::{self, UserFuncName},
    prelude::*,
};
use cranelift_jit::JITBuilder;
use cranelift_jit::JITModule;
use cranelift_module::{Linkage, Module};

use std::mem::{self};

#[allow(dead_code)]
fn create_module() -> JITModule {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    flag_builder
        .set("enable_llvm_abi_extensions", "true")
        .unwrap();
    let flags = settings::Flags::new(flag_builder);

    let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
        panic!("host machine is not supported: {}", msg);
    });
    let isa = isa_builder.finish(flags).unwrap();

    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let addr: *const u8 = numeric::add as *const u8;
    builder.symbol("add", addr);
    JITModule::new(builder)
}

#[allow(dead_code)]
fn call_add_func(a: Int32Array, b: Int32Array) -> Result<ArrayRef, ArrowError> {
    let mut module = create_module();
    let mut sig_call = module.make_signature();
    let pointer = module.target_config().pointer_type();
    let width_pointer = ir::Type::int((module.target_config().pointer_bits() * 2) as u16).unwrap();

    let mut sig_add = module.make_signature();
    sig_add.params.push(AbiParam::new(width_pointer));
    sig_add.params.push(AbiParam::new(width_pointer));
    sig_add.returns.push(AbiParam::new(pointer));
    let func_add = module
        .declare_function("add", Linkage::Import, &sig_add)
        .unwrap();

    sig_call.params.push(AbiParam::new(width_pointer));
    sig_call.params.push(AbiParam::new(width_pointer));
    sig_call.returns.push(AbiParam::new(pointer));
    let func_call = module
        .declare_function("call", Linkage::Local, &sig_call)
        .unwrap();

    let mut func_ctx = FunctionBuilderContext::new();
    let mut ctx = module.make_context();
    ctx.func.signature = sig_call;
    ctx.func.name = UserFuncName::user(0, func_call.as_u32());
    {
        let mut func_builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
        let ebb = func_builder.create_block();
        func_builder.switch_to_block(ebb);
        func_builder.append_block_params_for_function_params(ebb);
        let func_add_local = module.declare_func_in_func(func_add, func_builder.func);
        let lhs = func_builder.block_params(ebb)[0];
        let rhs = func_builder.block_params(ebb)[1];
        let call = func_builder.ins().call(func_add_local, &[lhs, rhs]);
        let value = {
            let result = func_builder.inst_results(call);
            assert_eq!(result.len(), 1);
            result[0]
        };
        func_builder.ins().return_(&[value]);
        func_builder.seal_all_blocks();
        func_builder.finalize();
    }
    module.define_function(func_call, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();
    let code_call = module.get_finalized_function(func_call);
    let call = unsafe {
        mem::transmute::<_, extern "C" fn(&dyn Datum, &dyn Datum) -> Result<ArrayRef, ArrowError>>(
            code_call,
        )
    };
    call(&a, &b)
}

#[cfg(test)]
mod tests {

    use arrow::array::{AsArray, Int32Array};

    use super::call_add_func;

    #[test]
    fn test_call_add_func() {
        let a1 = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let a2 = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let result = call_add_func(a1, a2).unwrap();
        let result_array: &Int32Array = result.as_primitive();
        assert_eq!(result_array.values(), &[2, 4, 6, 8, 10]);
    }
}
