// arrow native type expressing a rust type has the same in-memory representation as arrow.
// ArrayNativeType
// ArrayNativeTypeOp

#[cfg(test)]
pub mod tests {
    use arrow::datatypes::ArrowNativeTypeOp;

    #[test]
    fn test_native_type() {
        let a = 23_i32;
        let b = a.add_wrapping(24_i32);
        assert_eq!(b, 23 + 24);
    }
}
