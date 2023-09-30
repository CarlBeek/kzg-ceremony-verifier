mod g1;
mod g2;
mod scalar;

use self::{
    g1::{p1_affine_in_g1, p1_from_affine, p1_mult, p1s_mult_pippenger, p1_ifft_inplace},
    g2::{p2_affine_in_g2, p2_from_affine, p2_mult, p2_to_affine},
    scalar::{fr_from_scalar, fr_sub, fr_mul, fr_div, fr_exp, fr_zero, fr_one, scalar_from_fr, fr_from_u64},
};
use crate::{
    bls::{g1::p1_to_affine, g2::p2s_mult_pippenger},
    CeremonyError, ParseError, G1, G2,
};
use blst::{
    blst_core_verify_pk_in_g2, blst_final_exp, blst_fp12, blst_fr, blst_fr_add,
    blst_miller_loop, blst_p1, blst_p1_affine, blst_p1_generator, blst_p2_affine,
    blst_p2_affine_generator, blst_p2_generator, blst_scalar, blst_scalar_from_le_bytes,
    BLST_ERROR,
};
use rand::Rng;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::iter;

const CYPHER_SUITE: &'static str = "BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_POP_";
const PRIMITIVE_ROOT_OF_UNITY: u64 = 7;

/// Verifies that the given G1 points are valid.
///
/// Valid mean that they are uniquely encoded in compressed ZCash format and
/// represent curve points in the prime order subgroup.
///
/// # Errors
/// Returns an error if any of the `points` is not a compressed ZCash format
/// point on the curve, or if the point is not in the correct prime order
/// subgroup.
pub fn validate_g1(points: &[crate::G1]) -> Result<(), crate::CeremonyError> {
    points.into_par_iter().enumerate().try_for_each(|(i, &p)| {
        let p = blst_p1_affine::try_from(p)?;
        if !p1_affine_in_g1(&p) {
            return Err(CeremonyError::InvalidG1Power(
                i,
                ParseError::InvalidSubgroup,
            ));
        }
        Ok(())
    })
}

/// Verifies that the given G2 points are valid.
///
/// Valid mean that they are uniquely encoded in compressed ZCash format and
/// represent curve points in the prime order subgroup.
///
/// # Errors
/// Returns an error if any of the `points` is not a compressed ZCash format
/// point on the curve, or if the point is not in the correct prime order
/// subgroup.
pub fn validate_g2(points: &[crate::G2]) -> Result<(), crate::CeremonyError> {
    points.into_par_iter().enumerate().try_for_each(|(i, &p)| {
        let p = blst_p2_affine::try_from(p)?;
        if !p2_affine_in_g2(&p) {
            return Err(CeremonyError::InvalidG2Power(
                i,
                ParseError::InvalidSubgroup,
            ));
        }
        Ok(())
    })
}

/// Verify that the pubkey contains the contribution added
/// from `previous` to `tau`.
///
/// # Errors
/// Returns an error if any of the points is invalid, or if the pairing
/// check fails.
pub fn verify_pubkey(
    tau: crate::G1,
    previous: crate::G1,
    pubkey: crate::G2,
) -> Result<(), crate::CeremonyError> {
    let tau = blst_p1_affine::try_from(tau)?;
    let previous = blst_p1_affine::try_from(previous)?;
    let pubkey = blst_p2_affine::try_from(pubkey)?;

    unsafe {
        let g2 = *blst_p2_affine_generator();
        if pairing(&tau, &g2) != pairing(&previous, &pubkey) {
            return Err(CeremonyError::PubKeyPairingFailed);
        }
    }
    Ok(())
}

/// Verify that `powers` contains a sequence of powers of `tau`.
///
/// # Errors
/// Returns an error if any of the points is invalid, or if the points are
/// not a valid sequence of powers.
pub fn verify_g1(powers: &[crate::G1], tau: crate::G2) -> Result<(), crate::CeremonyError> {
    // Parse ZCash format
    let powers = powers
        .into_par_iter()
        .map(|p| blst_p1_affine::try_from(*p))
        .collect::<Result<Vec<_>, _>>()?;
    let tau = blst_p2_affine::try_from(tau)?;
    let tau = p2_from_affine(&tau);

    // Compute random linear combination
    let (factors, sum) = random_factors(powers.len() - 1);
    let g2 = unsafe { *blst_p2_generator() };

    let lhs_g1 = p1s_mult_pippenger(&powers[1..], &factors[..]);
    let lhs_g2 = p2_to_affine(&p2_mult(&g2, &sum));

    let rhs_g1 = p1s_mult_pippenger(&powers[..factors.len()], &factors[..]);
    let rhs_g2 = p2_to_affine(&p2_mult(&tau, &sum));

    // Check pairing
    if pairing(&lhs_g1, &lhs_g2) != pairing(&rhs_g1, &rhs_g2) {
        return Err(CeremonyError::G1PairingFailed);
    }

    Ok(())
}

/// Verify that `g1` and `g2` contain the same values.
///
/// # Errors
/// Returns an error if any of the points is invalid, if `g2` is not a valid
/// sequence of powers, or if `g1` and `g2` are sequences with different
/// exponents.
pub fn verify_g2(g1: &[crate::G1], g2: &[crate::G2]) -> Result<(), crate::CeremonyError> {
    assert!(g1.len() == g2.len());

    // Parse ZCash format
    let g1 = g1
        .into_par_iter()
        .map(|p| blst_p1_affine::try_from(*p))
        .collect::<Result<Vec<_>, _>>()?;

    let g2 = g2
        .into_par_iter()
        .map(|p| blst_p2_affine::try_from(*p))
        .collect::<Result<Vec<_>, _>>()?;

    // Compute random linear combination
    let (factors, sum) = random_factors(g2.len());
    let g1_generator = unsafe { *blst_p1_generator() };
    let g2_generator = unsafe { *blst_p2_generator() };

    let lhs_g1 = p1s_mult_pippenger(&g1, &factors[..]);
    let lhs_g2 = p2_to_affine(&p2_mult(&g2_generator, &sum));

    let rhs_g1 = p1_to_affine(&p1_mult(&g1_generator, &sum));
    let rhs_g2 = p2s_mult_pippenger(&g2, &factors[..]);

    // Check pairing
    if pairing(&lhs_g1, &lhs_g2) != pairing(&rhs_g1, &rhs_g2) {
        return Err(CeremonyError::G1PairingFailed);
    }

    Ok(())
}

pub fn verify_signature(sig: G1, message: &[u8], pk: G2) -> bool {
    let blst_pk = match blst_p2_affine::try_from(pk).ok() {
        Some(pk) => pk,
        _ => return false,
    };
    let blst_sig = match blst_p1_affine::try_from(sig).ok() {
        Some(sig) => sig,
        _ => return false,
    };
    let result = unsafe {
        blst_core_verify_pk_in_g2(
            &blst_pk,
            &blst_sig,
            true,
            message.as_ptr(),
            message.len(),
            CYPHER_SUITE.as_ptr(),
            CYPHER_SUITE.len(),
            [0; 0].as_ptr(),
            0,
        )
    };
    result == BLST_ERROR::BLST_SUCCESS
}

fn pairing(p: &blst_p1_affine, q: &blst_p2_affine) -> blst_fp12 {
    let mut tmp = blst_fp12::default();
    unsafe { blst_miller_loop(&mut tmp, q, p) };

    let mut out = blst_fp12::default();
    unsafe { blst_final_exp(&mut out, &tmp) };

    out
}

pub fn random_factors(n: usize) -> (Vec<blst_scalar>, blst_scalar) {
    let mut rng = rand::thread_rng();
    let mut entropy = [0u8; 32];

    let mut sum = blst_fr::default();
    let factors = iter::from_fn(|| {
        let mut scalar = blst_scalar::default();
        rng.fill(&mut entropy);
        unsafe {
            blst_scalar_from_le_bytes(&mut scalar, entropy.as_ptr(), entropy.len());
        }

        let r = fr_from_scalar(&scalar);
        unsafe { blst_fr_add(&mut sum, &sum, &r) };
        Some(scalar_from_fr(&r))
    })
    .take(n)
    .collect::<Vec<_>>();

    (factors, scalar_from_fr(&sum))
}


fn compute_root_of_unity(length: usize, primitive_root: blst_fr) -> blst_fr {
    let mod_minus_1 = fr_sub(&fr_zero(), &fr_one());
    let exponent = fr_div(&mod_minus_1, &fr_from_u64(length as u64));
    fr_exp(&primitive_root, &exponent)
}

fn compute_roots_of_unity(num_roots: usize, primitive_root: blst_fr) -> Vec<blst_fr> {
    let root_of_unity = compute_root_of_unity(num_roots, primitive_root);
    let mut roots = Vec::new();
    let mut current_root_of_unity = fr_one();
    for _ in 0..num_roots {
        roots.push(current_root_of_unity);
        current_root_of_unity = fr_mul(&current_root_of_unity, &root_of_unity);
    }
    roots
}


pub fn get_lagrange_g1(points: &[G1]) -> Result<Vec<G1>, CeremonyError> {
    let domain = compute_roots_of_unity(points.len(), fr_from_u64(PRIMITIVE_ROOT_OF_UNITY));
    let points_p1: Result<Vec<blst_p1>, ParseError> = points
        .par_iter()
        .map(|&p| blst_p1_affine::try_from(p).map(|p| p1_from_affine(&p)))
        .collect();

    let points_p1 = points_p1.map_err(|err| CeremonyError::from(err));

    p1_ifft_inplace(points_p1?, domain)
        .iter()
        .map(|p| G1::try_from(*p).map_err(|err| CeremonyError::from(err)))
        .collect::<Result<Vec<_>, _>>()
}


#[cfg(test)]
mod tests {
    use crate::bls::scalar::fr_from_lendian;

    use super::*;
    use super::super::hex_format::hex_str_to_bytes;

    #[test]
    fn test_compute_root_of_unity() {
        let primitive_root = fr_from_u64(PRIMITIVE_ROOT_OF_UNITY);
        let expected_root = fr_from_lendian(&hex_str_to_bytes("0x4b697f812fbd3549ffdee899a865080aadf40cac2181366b2ef1e9e298409b4f").unwrap());
        let root = compute_root_of_unity(256, primitive_root);
        assert_eq!(root, expected_root);
    }

    #[test]
    fn test_compute_roots_of_unity() {
        let primitive_root = fr_from_u64(PRIMITIVE_ROOT_OF_UNITY);
        let expected_roots = vec![
            fr_from_lendian(&hex_str_to_bytes("0x0100000000000000000000000000000000000000000000000000000000000000").unwrap()),
            fr_from_lendian(&hex_str_to_bytes("0x000000000000010000000376020003ecd0040376cecc518d0000000000000000").unwrap()),
            fr_from_lendian(&hex_str_to_bytes("0x00000000fffffffffe5bfeff02a4bd5305d8a10908d83933487d9d2953a7ed73").unwrap()),
            fr_from_lendian(&hex_str_to_bytes("0x01000000fffffefffe5bfb8900a4ba6734d39e93390be8a5477d9d2953a7ed73").unwrap()),
        ];

        let roots = compute_roots_of_unity(expected_roots.len(), primitive_root);
        assert_eq!(roots, expected_roots);
    }

    #[test]
    fn test_get_lagrange_g1() {
        // Values taken from py_spec https://github.com/ethereum/consensus-specs/blob/2ac06c10d31bb91f467214a1f13c0e55bd7ccef5/presets/minimal/trusted_setups/testing_trusted_setups.json
        let monomials = vec![
            G1(hex_str_to_bytes("0x97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb").unwrap()),
            G1(hex_str_to_bytes("0x854262641262cb9e056a8512808ea6864d903dbcad713fd6da8dddfa5ce40d85612c912063ace060ed8c4bf005bab839").unwrap()),
            G1(hex_str_to_bytes("0x86f708eee5ae0cf40be36993e760d9cb3b2371f22db3209947c5d21ea68e55186b30871c50bf11ef29e5248bf42d5678").unwrap()),
            G1(hex_str_to_bytes("0x94f9c0bafb23cbbf34a93a64243e3e0f934b57593651f3464de7dc174468123d9698f1b9dfa22bb5b6eb96eae002f29f").unwrap()),
        ];
        let expected_lagrange = vec![
            G1(hex_str_to_bytes("0x91131b2e3c1e5f0b51df8970e67080032f411571b66d301436c46f25bbfddf9ca16756430dc470bdb0d85b47fedcdbc1").unwrap()),
            G1(hex_str_to_bytes("0x934d35b2a46e169915718b77127b0d4efbacdad7fdde4593af7d21d37ebcb77fe6c8dde6b8a9537854d70ef1f291a585").unwrap()),
            G1(hex_str_to_bytes("0x9410ca1d0342fe7419f02194281df45e1c1ff42fd8b439de5644cc312815c21ddd2e3eeb63fb807cf837e68b76668bd5").unwrap()),
            G1(hex_str_to_bytes("0xb163df7e9baeb60f69b6ee5faa538c3a564b62eb8cde6a3616083c8cb2171eedd583c9143e7e916df59bf27da5e024e8").unwrap())
        ];

        let lagrange = get_lagrange_g1(&monomials).unwrap();
        assert_eq!(lagrange, expected_lagrange);
    }
}