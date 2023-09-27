use clap::{App, Arg};
use kzg_ceremony_verifier::{download_transcript, load_transcript, verify_transcript, save_individual_setups};

fn main() {
    // Define the command-line arguments
    let matches = App::new("Your CLI")
        .arg(
            Arg::with_name("url")
                .long("url")
                .takes_value(true)
                .default_value("https://seq.ceremony.ethereum.org/info/current_state")
                .help("The URL to download the transcript from"),
        )
        .arg(
            Arg::with_name("transcript_path")
                .long("transcript-path")
                .takes_value(true)
                .default_value("transcript.json")
                .help("The path to save the transcript to"),
        )
        .arg(
            Arg::with_name("output_folder")
                .long("output-folder")
                .takes_value(true)
                .default_value("./output_setups")
                .help("The folder to save the individual setups to"),
        )
        .arg(
            Arg::with_name("ceremony_sizes")
                .long("ceremony-sizes")
                .takes_value(true)
                .default_value("4096,65 8192,65 16384,65 32768,65")
                .help("The sizes of the ceremony in the format 'ceremony1_g1_size,ceremony1_g2_size ceremony2_g1_size,ceremony2_g2_size ...'"),
        )
        .get_matches();

    // Get the values of the command-line arguments
    let url = matches.value_of("url").unwrap();
    let transcript_path = matches.value_of("transcript_path").unwrap();
    let output_folder = matches.value_of("output_folder").unwrap();
    let ceremony_sizes: Vec<(usize, usize)> = matches.value_of("ceremony_sizes").unwrap().split_whitespace().map(|s| {
        let mut parts = s.split(',');
        let g1_size = parts.next().unwrap().parse::<usize>().unwrap();
        let g2_size = parts.next().unwrap().parse::<usize>().unwrap();
        (g1_size, g2_size)
    }).collect();

    // Download transcript
    let _ = download_transcript(transcript_path, url);

    // Load transcript
    let batch_transcript = load_transcript(transcript_path).unwrap();

    // Verify transcript
    let _ = verify_transcript(&batch_transcript, ceremony_sizes);

    // Save individual setups
    let _ = save_individual_setups(&batch_transcript, output_folder);
}