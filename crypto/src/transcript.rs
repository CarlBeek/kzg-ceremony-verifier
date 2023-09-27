use super::{CeremonyError, Powers, G1, G2, OutputJson};
use crate::signature::BlsSignature;
use crate::bls;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use rayon::iter::{ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator};

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Transcript {
    #[serde(flatten)]
    pub powers: Powers,

    pub witness: Witness,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Witness {
    #[serde(rename = "runningProducts")]
    pub products: Vec<G1>,

    #[serde(rename = "potPubkeys")]
    pub pubkeys: Vec<G2>,

    #[serde(rename = "blsSignatures")]
    pub signatures: Vec<BlsSignature>,
}

impl Transcript {
    /// Create a new transcript for a ceremony of a given size.
    ///
    /// # Panics
    ///
    /// There must be at least two g1 and two g2 points, and there must be at
    /// least as many g1 as g2 points.
    #[must_use]
    pub fn new(num_g1: usize, num_g2: usize) -> Self {
        assert!(num_g1 >= 2);
        assert!(num_g2 >= 2);
        assert!(num_g1 >= num_g2);
        Self {
            powers:  Powers::new(num_g1, num_g2),
            witness: Witness {
                products:   vec![G1::one()],
                pubkeys:    vec![G2::one()],
                signatures: vec![BlsSignature::empty()],
            },
        }
    }

    /// Returns the number of participants that contributed to this transcript.
    #[must_use]
    pub fn num_participants(&self) -> usize {
        self.witness.pubkeys.len() - 1
    }

    /// True if there is at least one contribution.
    #[must_use]
    pub fn has_entropy(&self) -> bool {
        self.num_participants() > 0
    }

    // Verifies the Powers of Tau are correctly constructed
    pub fn verify_powers(
        &self,
        num_g1: usize,
        num_g2: usize,
    ) -> Result<(), CeremonyError> {
        // Sanity checks on provided num_g1 and num_g2
        assert!(num_g1 >= 2);
        assert!(num_g2 >= 2);
        assert!(num_g1 >= num_g2);

        // Num powers checks
        // Note: num_g1_powers and num_g2_powers checked in TryFrom<PowersJson>
        if num_g1 != self.powers.g1.len() {
            return Err(CeremonyError::UnexpectedNumG1Powers(
                num_g1,
                self.powers.g1.len(),
            ));
        }
        if num_g2 != self.powers.g2.len() {
            return Err(CeremonyError::UnexpectedNumG2Powers(
                num_g2,
                self.powers.g2.len(),
            ));
        }

        // Point sanity checks (encoding and subgroup checks).
        bls::validate_g1(&self.powers.g1)?;
        bls::validate_g2(&self.powers.g2)?;

        // Non-zero checks
        if self
            .witness
            .pubkeys
            .par_iter()
            .any(|pubkey| *pubkey == G2::zero())
        {
            return Err(CeremonyError::ZeroPubkey);
        }
        // Verify powers are correctly constructed
        bls::verify_g1(&self.powers.g1, self.powers.g2[1])?;
        bls::verify_g2(&self.powers.g1[..self.powers.g2.len()], &self.powers.g2)?;

        Ok(())
    }

    pub fn verify_witnesses(
        &self,
    ) -> Result<(), CeremonyError> {
        // Sanity checks on num pubkeys & products
        if self.witness.products.len() != self.witness.pubkeys.len() {
            return Err(CeremonyError::WitnessLengthMismatch(
                self.witness.products.len(),
                self.witness.pubkeys.len(),
            ));
        }

        // Point sanity checks (encoding and subgroup checks).
        bls::validate_g1(&self.witness.products)?;
        bls::validate_g2(&self.witness.pubkeys)?;

        // Pairing check all pubkeys
        // TODO: figure out how to do this with some kind of batched pairings
        if self
            .witness
            .products
            .par_iter()
            .enumerate()
            .filter(|(i, _)| i >=  &self.witness.products.len())
            .any(|(i, product)|
                bls::verify_pubkey(
                    *product,
                    self.witness.products[i - 1],
                    self.witness.pubkeys[i],
                ).is_err()
            )
        {
            return Err(CeremonyError::PubKeyPairingFailed);
        }

        // Verify powers match final witness product
        if self.powers.g1[1] != self.witness.products[self.witness.products.len() - 1] {
            return Err(CeremonyError::G1ProductMismatch);
        }
        Ok(())
    }

    pub fn output_json_setup(&self, folder: &str) -> Result<(), CeremonyError> {
        let g1_lagrange = bls::get_lagrange_g1(&self.powers.g1)?;
        let json = OutputJson {
            g1_lagrange: g1_lagrange,
            g2_monomial: self.powers.g2.clone(),
        };

        let file_path = PathBuf::from(folder).join(format!("trusted_setup_{}.json", self.powers.g1.len()));
        let file = File::create(file_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &json)?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    // use crate::{
    //     CeremonyError::{
    //         G1PairingFailed, G2PairingFailed, InvalidG1Power, InvalidG2Power, PubKeyPairingFailed,
    //         UnexpectedNumG1Powers, UnexpectedNumG2Powers,
    //     },
    //     ParseError::InvalidSubgroup,
    // };
    // use ark_bls12_381::{Fr, G1Affine, G2Affine};
    // use ark_ec::{AffineCurve, ProjectiveCurve};
    // use hex_literal::hex;

    #[test]
    fn transcript_json() {
        let t = Transcript::new(4, 2);
        let json = serde_json::to_value(&t).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
            "numG1Powers": 4,
            "numG2Powers": 2,
            "powersOfTau": {
                "G1Powers": [
                "0x97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb",
                "0x97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb",
                "0x97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb",
                "0x97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"
                ],
                "G2Powers": [
                "0x93e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8",
                "0x93e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8",
                ],
            },
            "witness": {
                "runningProducts": [
                    "0x97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"
                ],
                "potPubkeys": [
                    "0x93e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8"
                ],
                "blsSignatures": [""],
            }
            })
        );
        let deser = serde_json::from_value::<Transcript>(json).unwrap();
        assert_eq!(deser, t);
    }
}
