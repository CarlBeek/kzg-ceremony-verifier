mod g1;
mod g2;
mod scalar;

use self::{
    g1::{p1_affine_in_g1, p1_from_affine, p1_add, p1_neg, p1_mult, p1s_mult_pippenger},
    g2::{p2_affine_in_g2, p2_from_affine, p2_mult, p2_to_affine},
    scalar::{fr_from_scalar, fr_sub, fr_mul, fr_div, fr_exp, fr_zero, fr_one, scalar_from_fr, fr_from_u64, fr_inv},
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

pub fn get_lagrange_g1(points: &[G1]) -> Result<Vec<G1>, CeremonyError> {
    let domain = compute_roots_of_unity(points.len(), fr_from_u64(PRIMITIVE_ROOT_OF_UNITY));
    let points_p1: Result<Vec<blst_p1>, ParseError> = points
        .par_iter()
        .map(|&p| blst_p1_affine::try_from(p).map(|p| p1_from_affine(&p)))
        .collect();

    let points_p1 = points_p1.map_err(|err| CeremonyError::from(err));

    let fft_output = fft(points_p1?, domain);

    let inv_length = fr_inv(&&fr_from_u64(points.len() as u64));

    let result: Result<Vec<G1>, CeremonyError> = fft_output
        .iter()
        .map(|point| {
            let res_point = p1_mult(&point, &scalar_from_fr(&inv_length));
            G1::try_from(p1_to_affine(&res_point)).map_err(|err| CeremonyError::from(err))
        })
        .collect();

    result
}

fn pairing(p: &blst_p1_affine, q: &blst_p2_affine) -> blst_fp12 {
    let mut tmp = blst_fp12::default();
    unsafe { blst_miller_loop(&mut tmp, q, p) };

    let mut out = blst_fp12::default();
    unsafe { blst_final_exp(&mut out, &tmp) };

    out
}

fn random_factors(n: usize) -> (Vec<blst_scalar>, blst_scalar) {
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


fn fft(vals: Vec<blst_p1>, domain: Vec<blst_fr>) -> Vec<blst_p1> {
    if vals.len() == 1 {
        return vals;
    }

    let l_vals: Vec<blst_p1> = vals.iter().step_by(2).cloned().collect();
    let r_vals: Vec<blst_p1> = vals.iter().skip(1).step_by(2).cloned().collect();
    
    let l_domain: Vec<blst_fr> = domain.iter().step_by(2).cloned().collect();
    
    let l = fft(l_vals, l_domain.clone());
    let r = fft(r_vals, l_domain);

    let mut o = vec![blst_p1::default(); vals.len()];
    for (i, (x, y)) in l.iter().zip(r.iter()).enumerate() {
        let y_times_root = p1_mult(&y, &scalar_from_fr(&domain[i]));
        
        o[i] = p1_add(&x, &y_times_root);
        o[i + l.len()] = p1_add(x, &p1_neg(&y_times_root));
    }
    o
}