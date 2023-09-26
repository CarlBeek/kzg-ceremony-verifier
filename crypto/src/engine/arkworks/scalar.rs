use ark_bls12_381::{Bls12_381, Fr};

pub fn fr_exp(a: &Fr, pow: u64) -> Fr {
    let mut result = Fr::zero();
    let mut base = *a;

    // Perform square and multiply algorithm
    let mut n = pow;
    while n > 0 {
        if n & 1 == 1 {
            // Multiply result by base
            result += &base;
        }
        // Square the base
        base = base.square();
        n >>= 1; // Right shift n by 1
    }
    result
}
