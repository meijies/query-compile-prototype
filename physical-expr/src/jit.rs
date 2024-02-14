use arrow::{
    array::{Array, ArrayRef, Datum, Int32Array},
    compute::kernels::numeric,
};
use cranelift::{codegen::ir::UserFuncName, prelude::*};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, Linkage, Module};
use std::{
    mem::{self, forget},
    sync::Arc,
};

pub(crate) struct JIT {
    func_ctx: FunctionBuilderContext,
    ctx: codegen::Context,
    data_description: DataDescription,
    module: JITModule,
}

extern "C" fn add_wrapper(
    lhs: *const Arc<dyn Datum>,
    rhs: *const Arc<dyn Datum>,
) -> *const Arc<dyn Array> {
    let lhs_ref = unsafe { Arc::from_raw(lhs) };
    let rhs_ref = unsafe { Arc::from_raw(rhs) };
    let res = numeric::add((*lhs_ref).as_ref(), (*rhs_ref).as_ref()).unwrap();
    return Arc::into_raw(Arc::new(res));
}

impl Default for JIT {
    fn default() -> Self {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();

        // build isa
        let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
            panic!("host machine is not supported: {}", msg);
        });
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();

        // build module
        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let addr: *const u8 = add_wrapper as *const u8;
        builder.symbol("add", addr);

        let module = JITModule::new(builder);
        Self {
            func_ctx: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            data_description: DataDescription::new(),
            module,
        }
    }
}

impl JIT {
    pub fn compile(&mut self) {
        let mut sig_call = self.module.make_signature();
        let pointer = self.module.target_config().pointer_type();

        let mut sig_add = self.module.make_signature();
        sig_add.params.push(AbiParam::new(pointer));
        sig_add.params.push(AbiParam::new(pointer));
        sig_add.returns.push(AbiParam::new(pointer));
        let func_add = self
            .module
            .declare_function("add", Linkage::Import, &sig_add)
            .unwrap();

        sig_call.params.push(AbiParam::new(pointer));
        sig_call.params.push(AbiParam::new(pointer));
        sig_call.returns.push(AbiParam::new(pointer));
        let func_call = self
            .module
            .declare_function("call", Linkage::Local, &sig_call)
            .unwrap();

        self.ctx.func.signature = sig_call;
        self.ctx.func.name = UserFuncName::user(0, func_call.as_u32());
        {
            let mut func_builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let ebb = func_builder.create_block();
            func_builder.switch_to_block(ebb);
            func_builder.append_block_params_for_function_params(ebb);
            let func_add_local = self
                .module
                .declare_func_in_func(func_add, &mut func_builder.func);
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
        self.module
            .define_function(func_call, &mut self.ctx)
            .unwrap();
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions().unwrap();
        let code_call = self.module.get_finalized_function(func_call);
        let call = unsafe {
            mem::transmute::<
                _,
                extern "C" fn(
                    *const Arc<dyn Datum>,
                    *const Arc<dyn Datum>,
                ) -> *const Arc<dyn Array>,
            >(code_call)
        };
        let a1: Arc<Arc<dyn Datum>> = Arc::new(Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])));
        let a2: Arc<Arc<dyn Datum>> = Arc::new(Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])));
        let result = call(Arc::into_raw(a1), Arc::into_raw(a2));

        let result = unsafe { Arc::from_raw(result) };
        println!("{:?}", (*result).clone());
    }
}

#[cfg(test)]
mod tests {

    use super::JIT;

    #[test]
    fn test_jit() {
        let mut jit = JIT::default();
        jit.compile();
    }
}
