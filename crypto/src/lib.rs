#![doc = include_str!("../Readme.md")]
#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]
#![allow(clippy::cast_lossless, clippy::module_name_repetitions)]
#![cfg_attr(any(test, feature = "bench"), allow(clippy::wildcard_imports))]

mod batch_transcript;
mod error;
mod group;
mod hex_format;
mod powers;
pub mod signature;
mod spinner;
mod transcript;
mod bls;

pub use crate::{
    batch_transcript::BatchTranscript,
    error::{CeremoniesError, CeremonyError, ErrorCode, ParseError},
    group::{F, G1, G2},
    powers::{OutputJson, Powers},
    signature::identity::Identity,
    spinner::create_spinner,
    transcript::Transcript,
};
