use kzg_ceremony_verifier::verify_with_file;

fn main() {
    let transcript_path = "transcript.json";
    let output_folder = "./output_setups";
    verify_with_file(transcript_path, output_folder).unwrap();
}