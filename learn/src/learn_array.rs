#[cfg(test)]
mod tests {

    use arrow::array::cast::as_primitive_array;
    use arrow::array::{Array, AsArray, Datum, Int32Array};

    #[test]
    fn create_scalar() {
        let scalar = Int32Array::new_scalar(1);
        let (array, is_scalar) = scalar.get();
        assert_eq!(is_scalar, true);
        let array: &Int32Array = array.as_primitive();
        assert_eq!(unsafe { array.value_unchecked(0) }, 1);
    }

    #[test]
    fn visit_scalar() {
        let array = Int32Array::from(vec![1, 2, 3, 4]);
        assert_eq!(unsafe {array.value_unchecked(0)}, 1);
        assert_eq!(array.value(1), 2);
    }

    #[test]
    fn create_array() {
        // create array directly
        let array = Int32Array::from(vec![1, 2, 3, 4]);
        assert_eq!(array.values(), &[1, 2, 3, 4]);
        let array = Int32Array::from(vec![Some(1), None, Some(2), None]);
        assert_eq!(array.null_count(), 2);
        let array = Int32Array::from_iter([1, 2, 3, 4]);
        assert_eq!(array.values(), &[1, 2, 3, 4]);
        let array = Int32Array::from_iter([Some(1), None, Some(3), None]);
        assert_eq!(array.values(), &[1, 0, 3, 0]);

        // create array from builder
        let mut builder = Int32Array::builder(5);
        builder.append_value(1);
        builder.append_value(2);
        builder.append_null();
        builder.append_slice(&[3, 4]);
        let array = builder.finish();
        assert_eq!(array.values(), &[1, 2, 0, 3, 4]);

        let array = array.slice(1, 2);
        assert_eq!(array.values(), &[2, 0]);
    }

    #[test]
    fn downcast() {
        let array = Int32Array::from(vec![1, 2, 3, 4]);
        let d1: &Int32Array = array.as_any().downcast_ref().unwrap();
        let d2: &Int32Array = as_primitive_array(&array);
        assert_eq!(d1.values(), d2.values());
    }
}
