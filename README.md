# KZG Ceremony Verifier

## Description

This is a tool for verifying the output of the Ethereum KZG Ceremony as well as transforming the output into a usable form for the various KZG libraries.

## Installation

Clone the repository and navigate to its directory:

```bash
git clone https://github.com/yourusername/trusted-setup-verifier.git
cd trusted-setup-verifier
```

Build the project:

```bash
cargo build --release
```

## Normal Usage

Run the executable from the target/release directory.

```bash
./target/release/your_cli
```

This will download the transcript from the sequencer, verify all the powers, check the contribution witnesses. If the above succseeds, it will save the ceremony output to `./output_setups/trusted_setup_#.json`.

## Advanced Usage

### Command-Line Arguments

The executable can be customised by supplying CLI arguments.

```bash
./target/release/your_cli [OPTIONS]
```

```text
--url

    Specifies the URL from which to download the transcript.
    Default: https://seq.ceremony.ethereum.org/info/current_state

--transcript-path

    Specifies the path to save the downloaded transcript.
    Default: transcript.json

--output-folder

    Specifies the folder to save the individual setups after verification.
    Default: ./output_setups

--ceremony-sizes

    Specifies the sizes of the ceremonies in the format 'ceremony1_g1_size,ceremony1_g2_size ceremony2_g1_size,ceremony2_g2_size ...'.
    Default: 4096,65 8192,65 16384,65 32768,65
```
