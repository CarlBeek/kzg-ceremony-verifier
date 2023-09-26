use kzg_ceremony_verifier::{load_transcript, verify_transcript, save_individual_setups};

fn main() {
    let transcript_path = "transcript.json";
    let output_folder = "./output_setups";
    let ceremony_sizes = vec![(4096, 65), (8192, 65), (16384, 65), (32768, 65)];
    let batch_transcript = load_transcript(transcript_path).unwrap();
    let _ = verify_transcript(&batch_transcript, ceremony_sizes);
    let _ = save_individual_setups(&batch_transcript, output_folder);
}