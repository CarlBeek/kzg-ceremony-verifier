use std::fs;

use crate::{
    signature::{identity::Identity, EcdsaSignature},
    CeremoniesError, create_spinner, Transcript,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::instrument;

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct BatchTranscript {
    pub transcripts:                  Vec<Transcript>,
    pub participant_ids:              Vec<Identity>,
    pub participant_ecdsa_signatures: Vec<EcdsaSignature>,
}

impl BatchTranscript {
    pub fn new<'a, I>(iter: I) -> Self
    where
        I: IntoIterator<Item = &'a (usize, usize)> + 'a,
    {
        Self {
            transcripts:                  iter
                .into_iter()
                .map(|(num_g1, num_g2)| Transcript::new(*num_g1, *num_g2))
                .collect(),
            participant_ids:              vec![Identity::None],
            participant_ecdsa_signatures: vec![EcdsaSignature::empty()],
        }
    }

    /// Returns the number of participants that contributed to this transcript.
    #[must_use]
    pub fn num_participants(&self) -> usize {
        self.participant_ids.len() - 1
    }


    pub fn verify_self(
        &self,
        sizes: Vec<(usize, usize)>,
    ) -> Result<(), CeremoniesError> {
        self.verify_powers(sizes)?;
        self.verify_witnesses()?;
        Ok(())
    }

    // Verifies the PoT of all the transcripts
    // given a vector of expected (num_g1, num_g2) points
    #[instrument(level = "info", skip_all, fields(n=self.transcripts.len()))]
    pub fn verify_powers(
        &self,
        sizes: Vec<(usize, usize)>,
    ) -> Result<(), CeremoniesError> {
        let spinner = create_spinner();
        spinner.set_message("Verifying Powers of Tau...");
        // Verify transcripts in parallel
        self.transcripts
            .par_iter()
            .zip(&sizes)
            .enumerate()
            .try_for_each(|(i, (transcript, (num_g1, num_g2)))| {
                transcript
                    .verify_powers(*num_g1, *num_g2)
                    .map_err(|e| CeremoniesError::InvalidCeremony(i, e))
            })?;
        spinner.finish_with_message("Powers of Tau verified");
        Ok(())
    }

    // Verifies witnesses are valid for all the transcripts
    #[instrument(level = "info", skip_all, fields(n=self.transcripts.len()))]
    pub fn verify_witnesses(
        &self,
    ) -> Result<(), CeremoniesError> {
        let spinner = create_spinner();
        spinner.set_message("Verifying contribution witnesses...");
        // Verify transcripts in parallel
        self.transcripts
            .par_iter()
            .enumerate()
            .try_for_each(|(i, transcript)| {
                transcript
                    .verify_witnesses()
                    .map_err(|e| CeremoniesError::InvalidCeremony(i, e))
            })?;
        spinner.finish_with_message("All contributions verified");
        Ok(())
    }


    pub fn output_json_setups(&self, folder: &str) -> Result<(), CeremoniesError> {
        // Create output folder if it doesn't exist
        fs::create_dir_all(&folder).unwrap();
        // Verify transcripts in parallel
        self.transcripts
            .par_iter()
            .enumerate()
            .try_for_each(|(i, transcript)| {
                transcript
                    .output_json_setup(folder)
                    .map_err(|e| CeremoniesError::InvalidCeremony(i, e))
            })?;
        Ok(())
    }
}
