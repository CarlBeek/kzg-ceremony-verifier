use crate::F;
use blst::{
    blst_fr, blst_fr_add, blst_fr_from_scalar, blst_fr_mul, blst_fr_sub, blst_fr_inverse, blst_fr_from_uint64, blst_lendian_from_scalar,
    blst_scalar, blst_scalar_from_fr, blst_scalar_from_lendian, blst_scalar_from_uint64, blst_uint64_from_fr, blst_fr_rshift,
};
use std::ops::Deref;
use zeroize::Zeroize;

#[allow(dead_code)] // Currently only used in tests
pub fn fr_add(a: &blst_fr, b: &blst_fr) -> blst_fr {
    let mut out = blst_fr::default();
    unsafe {
        blst_fr_add(&mut out, a, b);
    }
    out
}

pub fn fr_sub(a: &blst_fr, b: &blst_fr) -> blst_fr {
    let mut out = blst_fr::default();
    unsafe {
        blst_fr_sub(&mut out, a, b);
    }
    out
}

pub fn fr_mul(a: &blst_fr, b: &blst_fr) -> blst_fr {
    let mut out = blst_fr::default();
    unsafe {
        blst_fr_mul(&mut out, a, b);
    }
    out
}

pub fn fr_inv(a: &blst_fr) -> blst_fr {
    let mut out = blst_fr::default();
    unsafe {
        blst_fr_inverse(&mut out, a);
    }
    out
}

pub fn fr_div(a: &blst_fr, b: &blst_fr) -> blst_fr {
    let inv = fr_inv(b);
    fr_mul(a, &inv)
}

pub fn fr_is_odd(a: &blst_fr) -> bool {
    let lower_64 = u64_from_fr(a);
    lower_64 & 1 == 1
}

pub fn fr_exp(x: &blst_fr, n: &blst_fr) -> blst_fr {
    let zero = fr_from_u64(0);
    let one = fr_from_u64(1);
    if n == &zero {
        return one;
    }

    let mut n = n.clone();
    let mut x = x.clone();
    let mut y = fr_from_u64(1);
    while n != one {
        if fr_is_odd(&n) {
            unsafe {
                blst_fr_mul(&mut y, &x, &y);
                blst_fr_sub(&mut n, &n, &one);
            }
        }
        unsafe {
            blst_fr_mul(&mut x, &x, &x);
            blst_fr_rshift(&mut n, &n, 1);
        }
    }
    unsafe {
        blst_fr_mul(&mut y, &x, &y);
    }
    y
}

#[allow(dead_code)] // Currently only used in tests
pub fn fr_zero() -> blst_fr {
    fr_from_scalar(&scalar_from_u64(0u64))
}

pub fn fr_one() -> blst_fr {
    fr_from_scalar(&scalar_from_u64(1u64))
}

pub fn scalar_from_fr(a: &blst_fr) -> blst_scalar {
    let mut ret = blst_scalar::default();
    unsafe {
        blst_scalar_from_fr(&mut ret, a);
    }
    ret
}

pub fn fr_from_scalar(a: &blst_scalar) -> blst_fr {
    let mut ret = blst_fr::default();
    unsafe {
        blst_fr_from_scalar(&mut ret, a);
    }
    ret
}

pub fn scalar_from_u64(a: u64) -> blst_scalar {
    let mut scalar = blst_scalar::default();
    let input = [a, 0, 0, 0];
    unsafe {
        blst_scalar_from_uint64(&mut scalar, input.as_ptr());
    }
    scalar
}

pub fn fr_from_u64(a: u64) -> blst_fr {
    let mut fr = blst_fr::default();
    let input = [a, 0, 0, 0];
    unsafe {
        blst_fr_from_uint64(&mut fr, input.as_ptr());
    }
    fr
}

pub fn u64_from_fr(a: &blst_fr) -> u64 {
    let mut ret = [0u64; 4];
    unsafe {
        blst_uint64_from_fr(ret.as_mut_ptr(), a);
    }
    ret[0]
}

impl From<&F> for blst_scalar {
    fn from(n: &F) -> Self {
        let mut out = Self::default();
        unsafe {
            blst_scalar_from_lendian(&mut out, n.0.as_ptr());
        }
        out
    }
}

impl From<&F> for blst_fr {
    fn from(n: &F) -> Self {
        // TODO: Zeroize the temps
        let mut scalar = blst_scalar::default();
        let mut ret = Self::default();
        unsafe {
            blst_scalar_from_lendian(&mut scalar, n.0.as_ptr());
            blst_fr_from_scalar(&mut ret, &scalar);
        }
        ret
    }
}

impl From<&blst_fr> for F {
    fn from(n: &blst_fr) -> Self {
        let mut scalar = blst_scalar::default();
        let mut ret = [0u8; 32];
        unsafe {
            blst_scalar_from_fr(&mut scalar, n);
            blst_lendian_from_scalar(ret.as_mut_ptr(), &scalar);
        }
        Self(ret)
    }
}

pub struct Scalar(blst_scalar);

impl Deref for Scalar {
    type Target = blst_scalar;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Zeroize for Scalar {
    fn zeroize(&mut self) {
        self.0.b.zeroize();
    }
}

impl From<blst_scalar> for Scalar {
    fn from(s: blst_scalar) -> Self {
        Self(s)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_fr_is_odd() {
        for a in 0..128 {
            let  fr = fr_from_u64(a);
            let is_odd = fr_is_odd(&fr);

            assert_eq!(is_odd, a % 2 != 0);
        }
    }

    #[test]
    fn test_fr_exp() {
        for (base, exp, result) in [
            (2234234, 0, 1),
            (2, 1, 2),
            (2, 3, 8),
            (4294967261, 2, 18446743773061842121),
            (5, 27, 7450580596923828125),
        ] {
            let base = fr_from_u64(base);
            let exp = fr_from_u64(exp);
            let result = fr_from_u64(result);
            assert_eq!(fr_exp(&base, &exp), result);
        }
    }
}
