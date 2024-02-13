use std::sync::Arc;

pub struct FuncRegistry {}

pub struct MemoryPool {}

pub struct ExecContext {
    func_registry: FuncRegistry,
    memory_pool: MemoryPool,
}

pub type ExecContextRef = Arc<ExecContext>;

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

