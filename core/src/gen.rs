use crate::{jit::build::create_jit_module, native_opcall::NativeOpCall};
use cranelift::{codegen::ir::UserFuncName, prelude::*};
use cranelift_module::{FuncId, Linkage, Module};

pub trait ExprGen {
    fn gen(&self, ctx: &mut FuncGenContext) -> Value;
}

pub struct CodegenContext {
    _func_ctx: FunctionBuilderContext,
    pub module: cranelift_jit::JITModule,
    pub ctx: codegen::Context,
    pub ptype: Type,
}

impl Default for CodegenContext {
    fn default() -> Self {
        let module = create_jit_module();
        Self {
            _func_ctx: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            ptype: module.target_config().pointer_type(),
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
            func_id,
            builder: FunctionBuilder::new(&mut self.ctx.func, &mut self._func_ctx),
            module: &mut self.module,
            ptype: self.ptype,
        }
    }

    pub fn finalize(mut self, func_id: FuncId) -> *const u8 {
        self.module.define_function(func_id, &mut self.ctx).unwrap();
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions().unwrap();
        self.module.get_finalized_function(func_id)
    }
}

//TODO thread local for thread safe.
pub struct FuncGenContext<'a> {
    func_id: FuncId,
    pub builder: FunctionBuilder<'a>,
    pub module: &'a mut cranelift_jit::JITModule,
    pub ptype: Type,
}

impl<'a> FuncGenContext<'a> {
    pub fn mut_builder(&mut self) -> &mut FunctionBuilder<'a> {
        &mut self.builder
    }

    pub fn call_f64_add_wrapping(&mut self, lhs: Value, rhs: Value) -> Value {
        let op = NativeOpCall::Float64AddWrapping;
        self.call_binary(op, lhs, rhs)
    }

    pub fn call_f64_div_wrapping(&mut self, lhs: Value, rhs: Value) -> Value {
        let op = NativeOpCall::Float64DivWrapping;

        self.call_binary(op, lhs, rhs)
    }

    pub fn call_f64_lt(&mut self, lhs: Value, rhs: Value) -> Value {
        let op = NativeOpCall::Float64Lt;
        self.call_binary(op, lhs, rhs)
    }

    fn call_binary(&mut self, op: NativeOpCall, lhs: Value, rhs: Value) -> Value {
        let sig = op.signature(self.ptype);
        // FIXME this don't generate new func id during every call.
        let func_id = self
            .module
            .declare_function(op.name(), Linkage::Import, &sig)
            .unwrap();
        let func = self.module.declare_func_in_func(func_id, self.builder.func);
        let call = self.builder.ins().call(func, &[lhs, rhs]);
        let result = self.builder.inst_results(call);
        assert_eq!(result.len(), 1);
        result[0]
    }

    pub fn finalize(mut self, result: Value) -> FuncId {
        self.builder.ins().return_(&[result]);
        self.builder.seal_all_blocks();
        self.builder.finalize();
        self.func_id
    }
}
