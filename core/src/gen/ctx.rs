use std::collections::HashMap;

use crate::gen::build::CodegenContextBuilder;
use crate::gen::build::FuncRegister;
use crate::jit::native_opcall::NativeOpCall;
use cranelift::codegen::ir::stackslot::StackSize;
use cranelift::codegen::ir::StackSlot;
use cranelift::{
    codegen::{ir::UserFuncName, isa::CallConv},
    prelude::*,
};
use cranelift_module::{FuncId, Linkage, Module};

pub struct CodegenContext {
    pub(crate) func_ctx: FunctionBuilderContext,
    pub module: cranelift_jit::JITModule,
    pub ctx: codegen::Context,
    pub(crate) register_funcs: HashMap<&'static str, FuncRegister>,
}

impl CodegenContext {
    pub fn builder() -> CodegenContextBuilder {
        CodegenContextBuilder::new()
    }

    pub fn ptype(&self) -> Type {
        self.module.target_config().pointer_type()
    }

    pub fn finalize(mut self, func_id: FuncId) -> *const u8 {
        self.module.define_function(func_id, &mut self.ctx).unwrap();
        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions().unwrap();
        self.module.get_finalized_function(func_id)
    }

    pub fn create_func_gen_ctx(
        &mut self,
        name: &str,
        params: Vec<AbiParam>,
        returns: Vec<AbiParam>,
    ) -> FuncGenContext {
        let sig = Signature {
            call_conv: CallConv::Fast,
            params,
            returns,
        };

        let func_id = self
            .module
            .declare_function(name, Linkage::Local, &sig)
            .unwrap();

        self.ctx.func.signature = sig;
        self.ctx.func.name = UserFuncName::user(0, func_id.as_u32());
        let ptype = self.module.target_config().pointer_type();

        FuncGenContext {
            func_id,
            ptype,
            builder: FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx),
            module: &mut self.module,
            stack: None,
            stack_len: 0,
            stack_value_map: HashMap::new(),
        }
    }
}

struct StackValueInfo {
    name: &'static str,
    _type: Type,
    is_ref: bool,
    offset: StackSize,
}

//TODO thread local for thread safe.
pub struct FuncGenContext<'long, 'short> {
    func_id: FuncId,
    pub ptype: Type,
    pub builder: FunctionBuilder<'short>,
    module: &'long mut cranelift_jit::JITModule,
    stack: Option<StackSlot>,
    stack_len: StackSize,
    stack_value_map: HashMap<&'static str, StackValueInfo>,
}

impl<'long: 'short, 'short> FuncGenContext<'long, 'short> {
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

    pub fn finalize(mut self, results: &[Value]) -> FuncId {
        self.builder.ins().return_(results);
        self.builder.seal_all_blocks();
        self.builder.finalize();
        self.func_id
    }

    pub fn make_static_stack(&mut self, capacity: StackSize) {
        assert!(capacity > 0);
        self.stack = Some(
            self.builder
                .func
                .create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, capacity)),
        );
        self.stack_len = capacity;
    }

    pub fn stack_store_ref(&mut self, name: &'static str, value_type: Type, _ref: Value) {
        self.stack_store(name, true, value_type, _ref);
    }

    pub fn stack_store_value(&mut self, name: &'static str, _type: Type, value: Value) {
        self.stack_store(name, false, _type, value);
    }

    pub fn stack_store_ref_data(&mut self, name: &'static str, value: Value, offset: u32) {
        let info = self.stack_value_map.get(name).unwrap();
        assert!(info.is_ref);
        let offset = info._type.bytes() * offset;
        let _ref = self.stack_load_value(name);
        self.builder
            .ins()
            .store(MemFlags::new(), value, _ref, offset as i32);
    }

    fn stack_store(&mut self, name: &'static str, is_ref: bool, _type: Type, value: Value) {
        match self.stack_value_map.get(name) {
            Some(info) => {
                self.builder
                    .ins()
                    .stack_store(value, self.stack.unwrap(), info.offset as i32);
            }
            None => {
                let info = self
                    .stack_value_map
                    .insert(
                        name,
                        StackValueInfo {
                            name,
                            is_ref,
                            _type,
                            offset: self.stack_len,
                        },
                    )
                    .unwrap();
                self.builder
                    .ins()
                    .stack_store(value, self.stack.unwrap(), info.offset as i32);

                self.stack_len += 1;
            }
        }
    }

    pub fn stack_load_value(&mut self, name: &str) -> Value {
        let info = self.stack_value_map.get(name).unwrap();
        let _type = if info.is_ref { self.ptype } else { info._type };
        self.builder
            .ins()
            .stack_load(_type, self.stack.unwrap(), info.offset as i32)
    }

    pub fn stack_load_ref_data(&mut self, name: &str, offset: u32) -> Value {
        let info = self.stack_value_map.get(name).unwrap();
        let offset = info._type.bytes() * offset;
        let _type = info._type;

        assert!(info.is_ref);
        let _ref = self.stack_load_value(name);
        self.builder.ins().load(_type, MemFlags::new().with_aligned(),_ref, offset as i32)
    }

}
