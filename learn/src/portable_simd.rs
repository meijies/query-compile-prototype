use std::simd::f32x4;

fn first() {
    let a = f32x4::splat(4.0);
    let b = f32x4::from_array([1.0, 2.0, 3.0, 4.0]);
    println!("{:?}", a + b);
}

#[cfg(test)]
mod tests {
    use super::first;


    #[test]
    fn test_first() {
        first();
    }
}
