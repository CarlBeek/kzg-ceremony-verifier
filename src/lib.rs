use eyre::Result;
use std::{fs::File, path::Path};
use ark_serialize::Read;
use kzg_ceremony_crypto::{
    BLST,
    BatchTranscript,
    create_spinner,
};


fn read_json_file(string_path: &str) -> Result<String> {
    let path = Path::new(string_path);
    let mut file = File::open(path)
    .expect("error opening file");
    let mut content = String::new();
    file.read_to_string(&mut content)
    .expect("error reading file");
    Ok(content)
}

pub fn load_transcript(path: &str) -> Result<BatchTranscript> {
    let spinner = create_spinner();
    spinner.set_message("Loading transcript...");
    let json = read_json_file(path)?;
    let batch_transcript = serde_json::from_str::<BatchTranscript>(&json)?;
    spinner.finish_with_message("Transcript loaded.");
    Ok(batch_transcript)
}

pub fn verify_transcript(batch_transcript: &BatchTranscript, ceremony_sizes: Vec<(usize, usize)>) -> Result<()> {
    let result = batch_transcript.verify_self::<BLST>(ceremony_sizes);

    match result {
        Ok(_) => {
            println!("Transcript successfully verified!");
        },
        Err(err) => {
            println!("Transcript DID NOT verify!");
            println!("Specifically, the following error was encountered: {:?}", err);
        }
    }
    Ok(())
}


pub fn save_individual_setups(batch_transcript: &BatchTranscript, folder: &str) -> Result<()> {
    let spinner = create_spinner();
    spinner.set_message("Saving individual setups...");
    batch_transcript.output_json_setups::<BLST>(folder).unwrap();
    spinner.finish_with_message("Individual setups saved.");
    Ok(())
}
