use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use execution::context::ExecContextRef;

use crate::PhysicalOperator;

struct MemSourceScan {
    batch: RecordBatch,
}

impl MemSourceScan {
    fn new(batch: RecordBatch) -> Self {
        Self { batch }
    }
}

impl PhysicalOperator for MemSourceScan {
    fn schema(&self) -> SchemaRef {
        self.batch.schema().clone()
    }

    fn exec(&self, ctx: ExecContextRef) -> Result<RecordBatch, ()> {
        Ok(self.batch.clone())
    }
}
