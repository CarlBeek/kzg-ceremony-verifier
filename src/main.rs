use kzg_ceremony_verifier::verify_with_file;

fn main() {
    println!("Hello, wrapper-small-pot!");
    let transcript_path = "transcript.json";
    let output_folder = "output_setups";
    verify_with_file(transcript_path, output_folder).unwrap();
}