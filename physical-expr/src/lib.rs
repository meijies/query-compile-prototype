use std::sync::Arc;

use arrow::{array::{ArrayRef, Int64Array}, datatypes::{DataType, SchemaRef}, record_batch::RecordBatch};

pub mod expr;

#[derive(Clone, Debug)]
pub enum Datum {
    Array(ArrayRef),
    Scalar(ScalarValue),
}

impl Datum {
    pub fn as_ref(&self) ->  Arc<dyn arrow::array::Datum> {
        match self {
            Datum::Array(array) => Arc::new(array.clone()),
            Datum::Scalar(scalar_value) => match scalar_value {
                ScalarValue::Int64(value) => Arc::new(Int64Array::new_scalar(*value)),
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ScalarValue {
    Int64(i64),
}

pub trait PhysicalExpr: Send + Sync {
    fn output_type(&self, schema: SchemaRef) -> DataType;
    fn children(&self) -> Vec<Arc<dyn PhysicalExpr>>;
    // ArrayRef can represent both array and scalar value.
    fn eval(&self, batch: &RecordBatch) -> Result<Datum, ()>;
}

