use cranelift::prelude::Value;

mod build;
mod ctx;

pub use build::*;
pub use ctx::*;

pub trait ExprGen {
    fn gen(&self, ctx: &mut FuncGenContext) -> Value;
}
