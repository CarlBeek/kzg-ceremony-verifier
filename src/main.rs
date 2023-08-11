use std::time::Instant;
use kzg_ceremony_verifier::verify_with_file;

fn main() {
    println!("Hello, wrapper-small-pot!");
    let transcript_path = "transcript.json";
    println!("verify with file initialized");
    let start_verify = Instant::now();
    verify_with_file(transcript_path).unwrap();
    println!("verify time: {:?}", start_verify.elapsed());
}