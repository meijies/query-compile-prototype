use crate::{Datum, PhysicalExpr, ScalarValue};
use arrow::{
    datatypes::{DataType, SchemaRef},
    record_batch::RecordBatch,
};
use std::sync::Arc;

struct LiteralExpr {
    scalar: ScalarValue,
}

impl LiteralExpr {
    fn new(scalar: ScalarValue) -> Self {
        Self { scalar }
    }
}

impl PhysicalExpr for LiteralExpr {
    fn output_type(&self, _: SchemaRef) -> DataType {
        match self.scalar {
            ScalarValue::Int64(_) => DataType::Int64,
        }
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![]
    }

    fn eval(&self, _: &RecordBatch) -> Result<Datum, ()> {
        Ok(Datum::Scalar(self.scalar))
    }
}
