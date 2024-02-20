use std::collections::HashMap;

use crate::gen::ctx::CodegenContext;
use crate::jit::build::create_jit_module;
use crate::jit::native_opcall::NativeOpCall;
use cranelift::codegen::ir::Signature;
use cranelift::frontend::FunctionBuilderContext;
use cranelift_module::Module;

pub struct FuncRegister {
    pub(crate) name: &'static str,
    pub(crate) address: *const u8,
    pub(crate) sig: Option<Signature>,
}

pub struct CodegenContextBuilder {
    debug: bool,
    register_funcs: HashMap<&'static str, FuncRegister>,
}

impl Default for CodegenContextBuilder {
    fn default() -> Self {
        CodegenContextBuilder::new()
    }
}

impl CodegenContextBuilder {
    pub fn new() -> Self {
        let mut register_funcs = HashMap::new();
        for op in NativeOpCall::all_opcalls() {
            register_funcs.insert(
                op.name(),
                FuncRegister {
                    name: op.name(),
                    address: op.addr(),
                    sig: None,
                },
            );
        }
        Self {
            debug: false,
            register_funcs,
        }
    }

    pub fn debug(mut self) -> Self {
        self.debug = true;
        self
    }

    pub fn register_func(mut self, fun: FuncRegister) -> Self {
        let name = fun.name;
        let result = self.register_funcs.insert(name, fun);
        match result {
            Some(_) => self,
            None => panic!("can't register two functions with same name: {}", name),
        }
    }

    pub fn finish(mut self) -> CodegenContext {
        let module = create_jit_module(&self.register_funcs);
        let mut ctx = module.make_context();
        if self.debug {
            ctx.set_disasm(true);
        }
        CodegenContext {
            func_ctx: FunctionBuilderContext::new(),
            register_funcs: self.register_funcs,
            ctx,
            module,
        }
    }
}
