use arrow::datatypes::ArrowNativeTypeOp;
use cranelift::codegen::ir::Signature;
use cranelift::codegen::isa::CallConv;
use cranelift::prelude::*;

pub enum NativeOpCall {
    Float64AddWrapping,
    Float64DivWrapping,
    Float64Lt,
}

impl NativeOpCall {
    pub(crate) fn all_opcalls() -> &'static [NativeOpCall] {
        use NativeOpCall::*;
        &[Float64AddWrapping, Float64DivWrapping, Float64Lt]
    }

    pub(crate) fn signature(&self, pointer_type: Type) -> Signature {
        use NativeOpCall::*;
        match self {
            Float64AddWrapping | Float64DivWrapping => Signature {
                params: vec![AbiParam::new(types::F64), AbiParam::new(types::F64)],
                returns: vec![AbiParam::new(types::F64)],
                call_conv: CallConv::Fast,
            },
            Float64Lt => Signature {
                params: vec![AbiParam::new(types::F64), AbiParam::new(types::F64)],
                returns: vec![AbiParam::new(types::I8)],
                call_conv: CallConv::Fast,
            },
        }
    }

    pub(crate) fn name(&self) -> &str {
        use NativeOpCall::*;
        match self {
            Float64AddWrapping => "Float64AddWrapping",
            Float64DivWrapping => "Float64DivWrapping",
            Float64Lt => "Float64Lt",
        }
    }

    pub(crate) fn addr(&self) -> *const u8 {
        use NativeOpCall::*;
        match self {
            Float64AddWrapping => f64::add_wrapping as *const u8,
            Float64DivWrapping => f64::div_wrapping as *const u8,
            Float64Lt => lt_wrap as *const u8,
        }
    }
}

fn lt_wrap(a: f64, b: f64) -> bool {
    a < b
}
