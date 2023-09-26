use eyre::Result;
use std::{fs::File, path::Path};
use ark_serialize::Read;
use kzg_ceremony_crypto::{
    BLST,
    BatchTranscript,
    create_spinner,
};

pub fn verify_with_file(in_path: &str, out_folder: &str) -> Result<()> {
    let json = read_json_file(in_path)?;
    let batch_transcript = verify_with_string(json)?;
    println!("Transcript successfully verified!");
    batch_transcript.output_json_setups::<BLST>(out_folder).unwrap();
    Ok(())
}

pub fn verify_with_string(json: String) -> Result<BatchTranscript> {
    let spinner = create_spinner();
    spinner.set_message("Loading transcript...");

    let batch_transcript = serde_json::from_str::<BatchTranscript>(&json)
        .expect("BatchTranscript deserialization failed");

    spinner.finish_with_message("Transcript loaded.");

    let sizes = vec![(4096, 65), (8192, 65), (16384, 65), (32768, 65)];
    let result = batch_transcript.verify_self::<BLST>(sizes);

    // let is_valid = match result {
    //     Ok(()) => true,
    //     Err(error) => {
    //         println!("{:?}", error);
    //         false
    //     },
    // };
    Ok(batch_transcript)
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
