use crate::{
    signature::{identity::Identity, EcdsaSignature},
    CeremoniesError, Engine, create_spinner, Transcript,
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

    // Verifies an entire batch transcript (including all pairing checks)
    // given a vector of expected (num_g1, num_g2) points
    #[instrument(level = "info", skip_all, fields(n=self.transcripts.len()))]
    pub fn verify_self<E: Engine>(
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
                    .verify_self::<E>(*num_g1, *num_g2)
                    .map_err(|e| CeremoniesError::InvalidCeremony(i, e))
            })?;
        spinner.finish_with_message("Powers of Tau verified!");
        Ok(())
    }
}
