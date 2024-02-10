use std::{ops::Index, sync::Arc};

use arrow::{datatypes::{DataType, SchemaRef}, record_batch::RecordBatch};

use crate::PhysicalExpr;

use crate::Datum;

struct ColumnExpr {
    name: String,
    index: usize,
}

impl ColumnExpr {
    fn new(name: String, index: usize) -> Self {
        Self { name, index }
    }
}

impl PhysicalExpr for ColumnExpr {
    fn output_type(&self, schema: SchemaRef) -> DataType {
        schema.field(self.index).data_type().clone()
    }

    fn children(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        todo!()
    }

    fn eval(&self, batch: &RecordBatch) -> Result<Datum, ()> {
        Ok(Datum::Array(batch.index(&self.name).clone()))
    }
}
