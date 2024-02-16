use std::sync::Arc;

use crate::create_jit_module;
use cranelift::{codegen::ir::UserFuncName, prelude::*};
use cranelift_module::{Linkage, Module};

pub trait ExprGen {
    fn gen(&self, ctx: &mut FuncGenContext) -> Value;
}

pub struct CodegenContext {
    _func_ctx: FunctionBuilderContext,
    pub module: cranelift_jit::JITModule,
    pub ctx: codegen::Context,
}

impl Default for CodegenContext {
    fn default() -> Self {
        let module = create_jit_module();
        Self {
            _func_ctx: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            module,
        }
    }
}

impl CodegenContext {
    pub fn create_func_gen_ctx(
        &mut self,
        name: &str,
        params: Vec<AbiParam>,
        returns: Vec<AbiParam>,
    ) -> FuncGenContext {
        let sig = Signature {
            call_conv: self.module.target_config().default_call_conv,
            params,
            returns,
        };

        let func_id = self
            .module
            .declare_function(name, Linkage::Local, &sig)
            .unwrap();

        self.ctx.func.signature = sig;
        self.ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        FuncGenContext {
            builder: FunctionBuilder::new(&mut self.ctx.func, &mut self._func_ctx),
            module: &mut self.module,
        }
    }
}

//TODO thread local for thread safe.
pub struct FuncGenContext<'a> {
    pub builder: FunctionBuilder<'a>,
    pub module: &'a mut cranelift_jit::JITModule,
}
