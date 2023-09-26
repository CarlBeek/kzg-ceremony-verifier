//! Abstraction over the backend used for cryptographic operations.
//!
//! # To do
//!
//! * Add support for BLST backend.
//! * Better API for passing entropy (`Secret<_>` etc.)

#[cfg(feature = "arkworks")]
mod arkworks;
#[cfg(feature = "blst")]
mod blst;
mod both;

use crate::{CeremonyError, F, G1, G2};
pub use secrecy::Secret;

#[cfg(feature = "arkworks")]
pub use self::arkworks::Arkworks;
#[cfg(feature = "blst")]
pub use self::blst::BLST;
pub use self::both::Both;

pub type Entropy = Secret<[u8; 32]>;
pub type Tau = Secret<F>;

pub trait Engine {
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
    fn validate_g1(points: &[G1]) -> Result<(), CeremonyError>;

    /// Verifies that the given G2 points are valid.
    ///
    /// Valid mean that they are uniquely encoded in compressed ZCash format and
    /// represent curve points in the prime order subgroup.
    ///
    /// # Errors
    /// Returns an error if any of the `points` is not a compressed ZCash format
    /// point on the curve, or if the point is not in the correct prime order
    /// subgroup.
    fn validate_g2(points: &[G2]) -> Result<(), CeremonyError>;

    /// Verify that the pubkey contains the contribution added
    /// from `previous` to `tau`.
    ///
    /// # Errors
    /// Returns an error if any of the points is invalid, or if the pairing
    /// check fails.
    fn verify_pubkey(tau: G1, previous: G1, pubkey: G2) -> Result<(), CeremonyError>;

    /// Verify that `powers` contains a sequence of powers of `tau`.
    ///
    /// # Errors
    /// Returns an error if any of the points is invalid, or if the points are
    /// not a valid sequence of powers.
    fn verify_g1(powers: &[G1], tau: G2) -> Result<(), CeremonyError>;

    /// Verify that `g1` and `g2` contain the same values.
    ///
    /// # Errors
    /// Returns an error if any of the points is invalid, if `g2` is not a valid
    /// sequence of powers, or if `g1` and `g2` are sequences with different
    /// exponents.
    fn verify_g2(g1: &[G1], g2: &[G2]) -> Result<(), CeremonyError>;

    /// Derive a secret scalar $τ$ from the given entropy.
    fn generate_tau(entropy: &Entropy) -> Tau;

    /// Multiply elements of `powers` by powers of $τ$.
    ///
    /// # Errors
    /// Returns an error if any of `powers` is not a valid curve point.
    fn add_tau_g1(tau: &Tau, powers: &mut [G1]) -> Result<(), CeremonyError>;

    /// Multiply elements of `powers` by powers of $τ$.
    ///
    /// # Errors
    /// Returns an error if any of `powers` is not a valid curve point.
    fn add_tau_g2(tau: &Tau, powers: &mut [G2]) -> Result<(), CeremonyError>;

    /// Sign a message with `CYPHER_SUITE`, using $τ$ as the secret key.
    fn sign_message(tau: &Tau, message: &[u8]) -> Option<G1>;

    /// Verify a `CYPHER_SUITE` signature.
    fn verify_signature(sig: G1, message: &[u8], pk: G2) -> bool;

    fn get_lagrange_g1(points: &[G1]) -> Result<Vec<G1>, CeremonyError>;
}
