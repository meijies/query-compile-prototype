#[derive(Debug)]
pub enum ServerError {
    NotImplemented(String),
    NotSupported(String),
    ArgumentError(String),
}
