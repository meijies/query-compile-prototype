use std::sync::Arc;

use arrow::{compute::kernels::{cmp::lt, numeric::add}, datatypes::{DataType, SchemaRef}, record_batch::RecordBatch};

use crate::{jit::JIT, Codegen, Datum, PhysicalExpr};


enum Op {
    Add,
    Lt,
}

struct BinaryExpr {
    lhs: Arc<dyn PhysicalExpr>,
    op: Op,
    rhs: Arc<dyn PhysicalExpr>,
}

impl BinaryExpr {
    fn new(op: Op, lhs: Arc<dyn PhysicalExpr>, rhs: Arc<dyn PhysicalExpr>) -> Self {
        Self { lhs, op, rhs }
    }
}

impl PhysicalExpr for BinaryExpr {
    fn output_type(&self, _: SchemaRef) -> DataType {
        match self.op {
            Op::Add => DataType::Int64,
            Op::Lt => DataType::Boolean,
        }
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn eval(&self, batch: &RecordBatch) -> Result<Datum, ()> {
        let lhs = self.lhs.eval(batch).unwrap();
        let rhs = self.rhs.eval(batch).unwrap();
        match self.op {
            Op::Add => Ok(Datum::Array(add(&*lhs.as_ref(), &*rhs.as_ref()).unwrap())),
            Op::Lt => Ok(Datum::Array(Arc::new(lt(&*lhs.as_ref(), &*rhs.as_ref()).unwrap()))),
        }
    }
}

impl Codegen for BinaryExpr {

    fn gen(&self, jit: JIT) {
    
    }
}
