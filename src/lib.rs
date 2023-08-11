use eyre::Result;
use std::{fs::File, path::Path};
use ark_serialize::Read;
use kzg_ceremony_crypto::{
    BLST,
    BatchTranscript,
    create_spinner,
};

pub fn verify_with_file(in_path: &str) -> Result<()> {
    let json = read_json_file(in_path)?;
    let result = verify_with_string(json)?;
    Ok(println!("Verification is correct: {:?}", result))
}

pub fn verify_with_string(json: String) -> Result<bool> {
    let spinner = create_spinner();
    spinner.set_message("Parsing transcript...");

    let batch_transcript = serde_json::from_str::<BatchTranscript>(&json)
        .expect("BatchTranscript deserialization failed");

    spinner.finish_with_message("Transcript loaded.");

    let sizes = vec![(4096, 65), (8192, 65), (16384, 65), (32768, 65)];
    let result = batch_transcript.verify_self::<BLST>(sizes);

    let is_valid = match result {
        Ok(()) => true,
        Err(error) => {
            println!("{:?}", error);
            false
        },
    };
    Ok(is_valid)
}

fn read_json_file(string_path: &str) -> Result<String> {
    let path = Path::new(string_path);
    let mut file = File::open(path)
    .expect("error opening file");
    let mut content = String::new();
    file.read_to_string(&mut content)
    .expect("error reading file");
    Ok(content)
}
