use std::sync::Arc;

use crate::PhysicalOperator;
use arrow::array::AsArray;
use arrow::{
    compute::filter,
    datatypes::SchemaRef,
    record_batch::{RecordBatch, RecordBatchOptions},
};
use execution::context::ExecContext;
use physical_expr::PhysicalExpr;

struct FilterOperator {
    input: Arc<dyn PhysicalOperator>,
    predicate: Arc<dyn PhysicalExpr>,
}

impl FilterOperator {
    fn new(input: Arc<dyn PhysicalOperator>, predicate: Arc<dyn PhysicalExpr>) -> Self {
        Self { input, predicate }
    }
}

impl PhysicalOperator for FilterOperator {
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn exec(&self, ctx: std::sync::Arc<ExecContext>) -> Result<RecordBatch, ()> {
        let input = self.input.exec(ctx.clone()).unwrap();
        let predicate = self.predicate.eval(&input).unwrap();
        let bind = predicate.as_ref();
        let (predicate_array, _) = bind.get();
        let bool_array = predicate_array.as_boolean();
        let columns = input
            .columns()
            .iter()
            .map(|column| filter(column, bool_array).unwrap())
            .collect();
        let options = RecordBatchOptions::default()
            .with_row_count(Some(bool_array.values().count_set_bits()));
        Ok(RecordBatch::try_new_with_options(input.schema(), columns, &options).unwrap())
    }
}
