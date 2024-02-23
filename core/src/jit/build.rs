use cranelift::{codegen::isa::TargetIsa, prelude::*};
use cranelift_jit::{JITBuilder, JITModule};
use std::{collections::HashMap, sync::Arc};

use crate::gen::FuncRegister;

pub fn create_jit_module(funcs: &HashMap<&str, FuncRegister>) -> JITModule {
    let flags = build_flags();
    let isa = build_isa(flags);
    build_jit_module(isa, funcs)
}

fn build_flags() -> settings::Flags {
    let mut flag_builder = settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    flag_builder
        .set("enable_llvm_abi_extensions", "true")
        .unwrap();
    flag_builder.set("opt_level", "speed").unwrap();
    let flags = settings::Flags::new(flag_builder);
    assert!(!flags.use_colocated_libcalls());
    assert!(!flags.is_pic());
    flags
}

fn build_isa(flags: settings::Flags) -> Arc<dyn TargetIsa> {
    let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
        panic!("host machine is not supported: {}", msg);
    });

    isa_builder.finish(flags).unwrap()
}

fn build_jit_module(isa: Arc<dyn TargetIsa>, funcs: &HashMap<&str, FuncRegister>) -> JITModule {
    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    for (&k, v) in funcs {
        builder.symbol(k, v.address);
    }
    JITModule::new(builder)
}
