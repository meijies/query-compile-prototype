use std::{ops::Index, sync::Arc};

use arrow::array::ArrayRef;
use arrow::compute::kernels::numeric::add;
use arrow::{
    compute::{filter, kernels::cmp::lt},
    datatypes::{DataType, Field, Schema, SchemaRef},
    array::{AsArray, Int64Array},
    record_batch::{RecordBatch, RecordBatchOptions},
};

struct FuncRegistry {}

struct MemoryPool {}

struct ExecContext {
    func_registry: FuncRegistry,
    memory_pool: MemoryPool,
}

type ExecContextRef = Arc<ExecContext>;

impl ExecContext {
    fn new() -> Self {
        Self {
            func_registry: FuncRegistry {},
            memory_pool: MemoryPool {},
        }
    }

    fn as_ref(self) -> ExecContextRef {
        Arc::new(self)
    }
}

trait PhysicalExpr {
    fn output_type(&self, schema: SchemaRef) -> DataType;
    fn children(&self) -> Vec<Arc<dyn PhysicalExpr>>;
    // ArrayRef can represent both array and scalar value.
    fn eval(&self, batch: &RecordBatch) -> Result<Datum, ()>;
}

#[derive(Clone, Debug)]
enum Datum {
    Array(ArrayRef),
    Scalar(ScalarValue),
}

impl Datum {
    fn as_ref(&self) ->  Arc<dyn arrow::array::Datum> {
        match self {
            Datum::Array(array) => Arc::new(array.clone()),
            Datum::Scalar(scalar_value) => match scalar_value {
                ScalarValue::Int64(value) => Arc::new(Int64Array::new_scalar(*value)),
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum ScalarValue {
    Int64(i64),
}


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

trait PhysicalOperator {
    fn schema(&self) -> SchemaRef;

    fn exec(&self, ctx: ExecContextRef) -> Result<RecordBatch, ()>;
}

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
