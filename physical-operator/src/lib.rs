use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use execution::context::ExecContextRef;

pub mod operator;
pub mod source;

trait PhysicalOperator {
    fn schema(&self) -> SchemaRef;

    fn exec(&self, ctx: ExecContextRef) -> Result<RecordBatch, ()>;
}
