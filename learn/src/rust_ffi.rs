use libc::{c_int, size_t};

#[link(name = "snappy")]
extern "C" {
    fn snappy_max_compressed_length(source_length: size_t) -> size_t;
    fn snappy_compress(
        input: *const u8,
        input_length: size_t,
        compressed: *mut u8,
        compressed_length: *mut size_t,
    ) -> c_int;
    fn snappy_uncompress(
        compressed: *const u8,
        compressed_length: size_t,
        uncompressed: *mut u8,
        uncompressed_length: *mut size_t,
    ) -> c_int;

    fn snappy_uncompressed_length(
        compressed: *const u8,
        compressed_length: size_t,
        result: *mut size_t,
    ) -> c_int;
    fn snappy_validate_compressed_buffer(compressed: *const u8, compressed_length: size_t)
        -> c_int;
}

pub fn validate_compressed_buffer(src: &[u8]) -> bool {
    unsafe { snappy_validate_compressed_buffer(src.as_ptr(), src.len() as size_t) == 0 }
}

pub fn compress(src: &[u8]) -> Vec<u8> {
    unsafe {
        let srclen = src.len() as size_t;
        let psrc = src.as_ptr();

        let mut dstlen = snappy_max_compressed_length(srclen);
        let mut dst = Vec::with_capacity(dstlen as usize);
        let pdst = dst.as_mut_ptr();

        snappy_compress(psrc, srclen, pdst, &mut dstlen);
        dst.set_len(dstlen as usize);
        dst
    }
}

pub fn uncompress(src: &[u8]) -> Option<Vec<u8>> {
    unsafe {
        let srclen = src.len() as size_t;
        let psrc = src.as_ptr();

        let mut dstlen: size_t = 0;
        snappy_uncompressed_length(psrc, srclen, &mut dstlen);

        let mut dst = Vec::with_capacity(dstlen as usize);
        let pdst = dst.as_mut_ptr();

        if snappy_uncompress(psrc, srclen, pdst, &mut dstlen) == 0 {
            dst.set_len(dstlen as usize);

            Some(dst)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rust_ffi::{
        compress, snappy_max_compressed_length, uncompress, validate_compressed_buffer,
    };

    #[test]
    fn basic() {
        let x = unsafe { snappy_max_compressed_length(100) };
        println!("max compressed length of a 100 byte buffer : {}", x);
        assert_ne!(x, 0);
    }

    #[test]
    fn valid() {
        let d = vec![0xde, 0xad, 0xd0, 0x0d];
        let c: &[u8] = &compress(&d);
        assert!(validate_compressed_buffer(c));
        assert!(uncompress(c) == Some(d));
    }

    #[test]
    fn invalid() {
        let d = vec![0, 0, 0, 0];
        assert!(!validate_compressed_buffer(&d));
        assert!(uncompress(&d).is_none());
    }

    #[test]
    fn empty() {
        let d = vec![];
        assert!(!validate_compressed_buffer(&d));
        assert!(uncompress(&d).is_none());
        let c = compress(&d);
        assert!(validate_compressed_buffer(&c));
        assert!(uncompress(&c) == Some(d));
    }
}
