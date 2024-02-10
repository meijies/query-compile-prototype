use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use execution::context::ExecContextRef;

pub mod source;
pub mod operator;

trait PhysicalOperator {
    fn schema(&self) -> SchemaRef;

    fn exec(&self, ctx: ExecContextRef) -> Result<RecordBatch, ()>;
}
