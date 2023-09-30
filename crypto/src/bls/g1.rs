use crate::{ParseError, G1};
use blst::{
    blst_p1, blst_p1_affine, blst_p1_affine_compress, blst_p1_affine_in_g1, blst_p1_from_affine,
    blst_p1_add_or_double, blst_p1_cneg, blst_p1_mult, blst_p1_to_affine, blst_p1_uncompress, blst_p1s_mult_pippenger,
    blst_p1s_mult_pippenger_scratch_sizeof, blst_scalar, blst_fr, limb_t, BLST_ERROR,
};
use crate::bls::scalar::{fr_inv, fr_from_u64, scalar_from_fr};

use std::{mem::size_of, ptr};

impl TryFrom<G1> for blst_p1_affine {
    type Error = ParseError;

    fn try_from(g1: G1) -> Result<Self, Self::Error> {
        let mut p = Self::default();
        let result = unsafe { blst_p1_uncompress(&mut p, g1.0.as_ptr()) };
        if result != BLST_ERROR::BLST_SUCCESS {
            return match result {
                BLST_ERROR::BLST_POINT_NOT_IN_GROUP => Err(ParseError::InvalidSubgroup),
                _ => Err(ParseError::InvalidCompression),
            };
        }
        Ok(p)
    }
}

impl TryFrom<blst_p1_affine> for G1 {
    type Error = ParseError;

    fn try_from(g1: blst_p1_affine) -> Result<Self, Self::Error> {
        unsafe {
            let mut buffer = [0u8; 48];
            blst_p1_affine_compress(buffer.as_mut_ptr(), &g1);
            Ok(Self(buffer))
        }
    }
}

impl TryFrom<blst_p1> for G1 {
    type Error = ParseError;

    fn try_from(g1: blst_p1) -> Result<Self, Self::Error> {
        Self::try_from(p1_to_affine(&g1))
    }
}

pub fn p1_from_affine(a: &blst_p1_affine) -> blst_p1 {
    unsafe {
        let mut p = blst_p1::default();
        blst_p1_from_affine(&mut p, a);
        p
    }
}

pub fn p1_to_affine(a: &blst_p1) -> blst_p1_affine {
    unsafe {
        let mut p = blst_p1_affine::default();
        blst_p1_to_affine(&mut p, a);
        p
    }
}

pub fn p1_add(a: &blst_p1, b: &blst_p1) -> blst_p1 {
    unsafe {
        let mut out = blst_p1::default();
        blst_p1_add_or_double(&mut out, a, b);
        out
    }
}

pub fn p1_neg(a: &blst_p1) -> blst_p1 {
    unsafe {
        let mut out = a.clone();
        blst_p1_cneg(&mut out, true);
        out
    }
}

pub fn p1_sub(a: &blst_p1, b: &blst_p1) -> blst_p1 {
    p1_add(&a, &p1_neg(&b))
}

pub fn p1_mult(p: &blst_p1, s: &blst_scalar) -> blst_p1 {
    unsafe {
        let mut out = blst_p1::default();
        blst_p1_mult(&mut out, p, s.b.as_ptr(), 255);
        out
    }
}

pub fn p1_affine_in_g1(p: &blst_p1_affine) -> bool {
    unsafe { blst_p1_affine_in_g1(p) }
}

pub fn p1s_mult_pippenger(bases: &[blst_p1_affine], scalars: &[blst_scalar]) -> blst_p1_affine {
    assert_eq!(bases.len(), scalars.len());
    if bases.is_empty() {
        // NOTE: Without this special case the `blst_p1s_mult_pippenger` will
        // SIGSEGV.
        return blst_p1_affine::default();
    }
    if bases.len() == 1 {
        // NOTE: Without this special case the `blst_p1s_mult_pippenger` will
        // SIGSEGV.
        let base = p1_from_affine(&bases[0]);
        let result = p1_mult(&base, &scalars[0]);
        return p1_to_affine(&result);
    }

    let npoints = bases.len();

    // Get vec of pointers to bases
    let points_ptrs = [bases.as_ptr(), ptr::null()];

    // Get vec of pointers to scalars
    assert_eq!(size_of::<blst_scalar>(), 32);
    let scalar_ptrs = [scalars.as_ptr(), ptr::null()];

    let scratch_size = unsafe { blst_p1s_mult_pippenger_scratch_sizeof(npoints) };
    let mut scratch = vec![limb_t::default(); scratch_size / size_of::<limb_t>()];
    let mut msm_result = blst_p1::default();
    let mut ret = blst_p1_affine::default();
    unsafe {
        blst_p1s_mult_pippenger(
            &mut msm_result,
            points_ptrs.as_ptr(),
            npoints,
            scalar_ptrs.as_ptr().cast(),
            256,
            scratch.as_mut_ptr(),
        );
        blst_p1_to_affine(&mut ret, &msm_result);
    }

    ret
}


fn fft_g1_fast(
    out: &mut [blst_p1],
    inp: &[blst_p1],
    stride: usize,
    roots: &[blst_fr],
    roots_stride: usize,
    n: usize,
) {
    let half = n / 2;
    if half > 0 {
        fft_g1_fast(&mut out[0..half], &inp[0..], stride * 2, &roots[0..], roots_stride * 2, half);
        fft_g1_fast(&mut out[half..n], &inp[stride..], stride * 2, &roots[0..], roots_stride * 2, half);
        
        for i in 0..half {
            let y_times_root = p1_mult(&out[i + half], &scalar_from_fr(&roots[i * roots_stride]));
            
            out[i + half] = p1_sub(&out[i], &y_times_root);
            out[i] = p1_add(&out[i], &y_times_root);
        }
    } else {
        out[0] = inp[0];
    }
}


pub fn p1_fft_inplace(vals: Vec<blst_p1>, roots: Vec<blst_fr>) -> Vec<blst_p1> {
    let mut ret = vals.clone();
    fft_g1_fast(&mut ret, &vals, 1, &roots, 1, vals.len());
    ret
}

pub fn p1_ifft_inplace(vals: Vec<blst_p1>, roots: Vec<blst_fr>) -> Vec<blst_p1> {
    let inv_len = scalar_from_fr(&fr_inv(&fr_from_u64(vals.len() as u64)));
    let reversed_roots = reverse_roots(&roots);
    let fft_points = p1_fft_inplace(vals, reversed_roots);
    fft_points.iter().map(|p1| p1_mult(&p1, &inv_len)).collect()
}

pub fn p1_fft_clone(vals: Vec<blst_p1>, roots: Vec<blst_fr>) -> Vec<blst_p1> {
    if vals.len() == 1 {
        return vals;
    }

    let l_vals: Vec<blst_p1> = vals.iter().step_by(2).cloned().collect();
    let r_vals: Vec<blst_p1> = vals.iter().skip(1).step_by(2).cloned().collect();

    let l_roots: Vec<blst_fr> = roots.iter().step_by(2).cloned().collect();

    let l = p1_fft_clone(l_vals, l_roots.clone());
    let r = p1_fft_clone(r_vals, l_roots);

    let mut o = vec![blst_p1::default(); vals.len()];
    for (i, (x, y)) in l.iter().zip(r.iter()).enumerate() {
        let y_times_root = p1_mult(&y, &scalar_from_fr(&roots[i]));
    
        o[i] = p1_add(&x, &y_times_root);
        o[i + l.len()] = p1_sub(&x, &y_times_root);
    }
    o
}

pub fn p1_ifft_clone(vals: Vec<blst_p1>, roots: Vec<blst_fr>) -> Vec<blst_p1> {
    let inv_length = scalar_from_fr(&fr_inv(&fr_from_u64(vals.len() as u64)));
    let reversed_roots = reverse_roots(&roots);
    let fft_points = p1_fft_clone(vals, reversed_roots);
    fft_points.iter().map(|p1| p1_mult(&p1, &inv_length)).collect()
}

pub fn fft_g1_slow(
    ret: &mut [blst_p1],
    data: &[blst_p1],
    stride: usize,
    roots: &[blst_fr],
    roots_stride: usize,
) {
    for i in 0..data.len() {
        // Evaluate first member at 1
        ret[i] = p1_mult(&data[0], &scalar_from_fr(&roots[0]));

        // Evaluate the rest of members using a step of (i * J) % data.len() over the roots
        // This distributes the roots over correct x^n members and saves on multiplication
        for j in 1..data.len() {
            let v = p1_mult(&data[j * stride], &scalar_from_fr(&(&roots[((i * j) % data.len()) * roots_stride])));
            ret[i] = p1_add(&ret[i], &v);
        }
    }
}

pub fn p1_fft_slow(vals: Vec<blst_p1>, roots: Vec<blst_fr>) -> Vec<blst_p1> {
    let mut ret = vals.clone();
    fft_g1_slow(&mut ret, &vals, 1, &roots, 1);
    ret
}

fn reverse_roots(roots: &Vec<blst_fr>) -> Vec<blst_fr> {
    let mut roots = roots.clone();
    let (first, rest) = roots.split_at_mut(1);
    rest.reverse();
    [first, rest].concat()
}

pub fn p1_ifft_slow(vals: Vec<blst_p1>, roots: Vec<blst_fr>) -> Vec<blst_p1> {
    let inv_len = scalar_from_fr(&fr_inv(&fr_from_u64(vals.len() as u64)));
    let reversed_roots = reverse_roots(&roots);
    let fft_points = p1_fft_slow(vals, reversed_roots);
    fft_points.iter().map(|p1| p1_mult(&p1, &inv_len)).collect()
}

#[cfg(test)]
mod tests {
    use crate::bls::scalar::scalar_from_u64;

    use super::{
        super::scalar::{fr_add, fr_from_scalar, fr_mul, fr_zero, scalar_from_fr},
        *,
    };
    use blst::blst_scalar_from_lendian;
    use proptest::{arbitrary::any, collection::vec as arb_vec, proptest, strategy::Strategy};
    use ruint::{aliases::U256, uint};


    #[test]
    fn test_p1_sub() {
        let one = p1_from_affine(&blst_p1_affine::try_from(G1::one()).unwrap());
        let two = p1_mult(&one, &scalar_from_u64(2));
        assert_eq!(p1_sub(&two, &one), one);
    }

    #[test]
    fn test_p1_sub_dumb() {
        let zero = blst_p1_affine::try_from(G1::zero()).unwrap();
        let points = get_random_p1s(1024);
        for p in points {
            assert_eq!(zero, p1_to_affine(&p1_sub(&p, &p)));
        }
    }

    pub fn arb_scalar() -> impl Strategy<Value = blst_scalar> {
        any::<U256>().prop_map(|mut n| {
            n %= uint!(52435875175126190479447740508185965837690552500527637822603658699938581184513_U256);
            let mut scalar = blst_scalar::default();
            unsafe {
                blst_scalar_from_lendian(&mut scalar, n.as_le_slice().as_ptr());
            }
            scalar
        })
    }

    #[test]
    fn test_p1s_mult_pippenger() {
        const SIZES: [usize; 7] = [0, 1, 2, 3, 4, 5, 100];
        for size in SIZES {
            proptest!(|(base in arb_vec(arb_scalar(), size), scalars in arb_vec(arb_scalar(), size))| {
                // Compute expected value
                let sum = base.iter().zip(scalars.iter()).fold(fr_zero(), |a, (l, r)| {
                    let product = fr_mul(&fr_from_scalar(l), &fr_from_scalar(r));
                    fr_add(&a, &product)
                });
                let sum = scalar_from_fr(&sum);
                let one = p1_from_affine(&blst_p1_affine::try_from(G1::one()).unwrap());
                let expected = p1_mult(&one, &sum);

                // Compute base points
                let base = base.iter().map(|s| {
                    p1_to_affine(&p1_mult(&one, s))
                }).collect::<Vec<_>>();

                // Compute dot product
                let result = p1s_mult_pippenger(base.as_slice(), scalars.as_slice());

                // Check result
                assert_eq!(p1_from_affine(&result), expected);
            });
        }
    }

    fn get_random_p1s (n: usize) -> Vec<blst_p1> {
        let (rand_factors, _) = crate::bls::random_factors(n);
        let rand_p1s = rand_factors.iter().map(|s| {
            p1_mult(&p1_from_affine(&blst_p1_affine::try_from(G1::one()).unwrap()), s)
        }).collect();
        rand_p1s
    }

    #[test]
    fn test_p1_fft_inplace_roundtrip() {
        const N: usize = 4;
        let original_p1s = get_random_p1s(N);
        let roots = crate::bls::compute_roots_of_unity(N, fr_from_u64(crate::bls::PRIMITIVE_ROOT_OF_UNITY));
        let fft_points = p1_fft_inplace(original_p1s.clone(), roots.clone());
        let ifft_points = p1_ifft_inplace(fft_points, roots);

        let original_g1s: Vec<G1> = original_p1s.iter().map(|p1| {
            G1::try_from(p1_to_affine(p1)).unwrap()
        }).collect();

        let ifft_g1s: Vec<G1> = ifft_points.iter().map(|p1| {
            G1::try_from(p1_to_affine(p1)).unwrap()
        }).collect();
        assert_eq!(original_g1s, ifft_g1s);
    }

    #[test]
    fn test_p1_fft_clone_roundtrip() {
        const N: usize = 4;
        let original_p1s = get_random_p1s(N);
        let roots = crate::bls::compute_roots_of_unity(N, fr_from_u64(crate::bls::PRIMITIVE_ROOT_OF_UNITY));
        let fft_points = p1_fft_clone(original_p1s.clone(), roots.clone());
        let ifft_points = p1_ifft_clone(fft_points, roots);

        let original_g1s: Vec<G1> = original_p1s.iter().map(|p1| {
            G1::try_from(p1_to_affine(p1)).unwrap()
        }).collect();

        let ifft_g1s: Vec<G1> = ifft_points.iter().map(|p1| {
            G1::try_from(p1_to_affine(p1)).unwrap()
        }).collect();
        assert_eq!(original_g1s, ifft_g1s);
    }

    #[test]
    fn test_p1_fft_slow_roundtrip() {
        const N: usize = 4;
        let original_p1s = get_random_p1s(N);
        let roots = crate::bls::compute_roots_of_unity(N, fr_from_u64(crate::bls::PRIMITIVE_ROOT_OF_UNITY));
        let fft_points = p1_fft_slow(original_p1s.clone(), roots.clone());
        let ifft_points = p1_ifft_slow(fft_points, roots);

        let original_g1s: Vec<G1> = original_p1s.iter().map(|p1| {
            G1::try_from(p1_to_affine(p1)).unwrap()
        }).collect();

        let ifft_g1s: Vec<G1> = ifft_points.iter().map(|p1| {
            G1::try_from(p1_to_affine(p1)).unwrap()
        }).collect();
        assert_eq!(original_g1s, ifft_g1s);
    }

    #[test]
    fn test_p1_fft_equality_inplace_clone() {
        const N: usize = 8;
        let original_p1s = get_random_p1s(N);
        let roots = crate::bls::compute_roots_of_unity(N, fr_from_u64(crate::bls::PRIMITIVE_ROOT_OF_UNITY));
        let clone_fft_points = p1_fft_clone(original_p1s.clone(), roots.clone());
        let inplace_fft_points = p1_fft_inplace(original_p1s.clone(), roots.clone());

        let clone_fft_G1s: Vec<G1> = clone_fft_points.iter().map(|p1| {
            G1::try_from(p1_to_affine(p1)).unwrap()
        }).collect();
        let inplace_fft_G1s: Vec<G1> = inplace_fft_points.iter().map(|p1| {
            G1::try_from(p1_to_affine(p1)).unwrap()
        }).collect();

        assert_eq!(clone_fft_G1s, inplace_fft_G1s);
    }


    #[test]
    fn test_p1_fft_equality_inplace_slow() {
        const N: usize = 8;
        let original_p1s = get_random_p1s(N);
        let roots = crate::bls::compute_roots_of_unity(N, fr_from_u64(crate::bls::PRIMITIVE_ROOT_OF_UNITY));
        let slow_fft_points = p1_fft_slow(original_p1s.clone(), roots.clone());
        let inplace_fft_points = p1_fft_inplace(original_p1s.clone(), roots.clone());

        let slow_fft_G1s: Vec<G1> = slow_fft_points.iter().map(|p1| {
            G1::try_from(p1_to_affine(p1)).unwrap()
        }).collect();
        let inplace_fft_G1s: Vec<G1> = inplace_fft_points.iter().map(|p1| {
            G1::try_from(p1_to_affine(p1)).unwrap()
        }).collect();

        assert_eq!(slow_fft_G1s, inplace_fft_G1s);
    }
}
