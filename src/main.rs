use std::{ops::Index, sync::Arc};

use arrow::array::ArrayRef;
use arrow::compute::kernels::numeric::add;
use arrow::{
    compute::{filter, kernels::cmp::lt},
    datatypes::{DataType, Field, Schema, SchemaRef},
    array::{AsArray, Int64Array},
    record_batch::{RecordBatch, RecordBatchOptions},
};

fn main() {
    let ctx = ExecContext::new().as_ref();

    // create schema with one field num
    let array = Int64Array::from(vec![1, 2, 3, 4, 5]);
    let schema = Schema::new(vec![Field::new("num", DataType::Int64, false)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap();

    // create mem source with one field num
    let source = MemSourceScan::new(batch);
    let filter = FilterOperator::new(
        Arc::new(source),
        Arc::new(BinaryExpr::new(
            Op::Lt,
            Arc::new(ColumnExpr::new(String::from("num"), 0)),
            Arc::new(LiteralExpr::new(ScalarValue::Int64(3))),
        )),
    );

    let result = filter.exec(ctx).unwrap();
    println! {"{:?}", result};
}
