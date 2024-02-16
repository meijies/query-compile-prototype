use arrow::{
    array::{Array, AsArray, Datum, Int32Array},
    compute::kernels::numeric,
};
use cranelift::{
    codegen::ir::UserFuncName,
    prelude::*,
};
use cranelift_jit::JITBuilder;
use cranelift_jit::JITModule;
use cranelift_module::{Linkage, Module};

use std::{
    mem::{self},
    sync::Arc,
};

extern "C" fn add_wrapper(
    lhs: *const Arc<dyn Datum>,
    rhs: *const Arc<dyn Datum>,
) -> *const Arc<dyn Array> {
    let lhs_ref = unsafe { Arc::from_raw(lhs) };
    let rhs_ref = unsafe { Arc::from_raw(rhs) };
    let res = numeric::add((*lhs_ref).as_ref(), (*rhs_ref).as_ref()).unwrap();
    Arc::into_raw(Arc::new(res))
}

fn create_module() -> JITModule {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let flags = settings::Flags::new(flag_builder);

    let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
        panic!("host machine is not supported: {}", msg);
    });
    let isa = isa_builder.finish(flags).unwrap();

    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let addr: *const u8 = add_wrapper as *const u8;
    builder.symbol("add", addr);
    JITModule::new(builder)
}

fn call_add_func() {
    let mut module = create_module();
    let mut sig_call = module.make_signature();
    let pointer = module.target_config().pointer_type();

    let mut sig_add = module.make_signature();
    sig_add.params.push(AbiParam::new(pointer));
    sig_add.params.push(AbiParam::new(pointer));
    sig_add.returns.push(AbiParam::new(pointer));
    let func_add = module
        .declare_function("add", Linkage::Import, &sig_add)
        .unwrap();

    sig_call.params.push(AbiParam::new(pointer));
    sig_call.params.push(AbiParam::new(pointer));
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
        let func_add_local = module.declare_func_in_func(func_add, &mut func_builder.func);
        let lhs = func_builder.block_params(ebb)[0];
        let rhs = func_builder.block_params(ebb)[1];
        let call = func_builder.ins().call(func_add_local, &[lhs, rhs]);
        let value = {
            let result = func_builder.inst_results(call);
            assert_eq!(result.len(), 1);
            result[0].clone()
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
        mem::transmute::<
            _,
            extern "C" fn(*const Arc<dyn Datum>, *const Arc<dyn Datum>) -> *const Arc<dyn Array>,
        >(code_call)
    };
    let a1: Arc<Arc<dyn Datum>> = Arc::new(Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])));
    let a2: Arc<Arc<dyn Datum>> = Arc::new(Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])));
    let result = call(Arc::into_raw(a1), Arc::into_raw(a2));

    let result = unsafe { Arc::from_raw(result) };
    let result_array : &Int32Array = result.as_primitive();
    assert_eq!(result_array.values(),  &[2, 4, 6, 8, 10]);
}

#[cfg(test)]
mod tests {
    
    use super::call_add_func;

    #[test]
    fn test_call_add_func() {
        call_add_func();
    }
}
