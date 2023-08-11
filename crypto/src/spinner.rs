use indicatif::{ProgressBar, ProgressStyle};

pub fn create_spinner() -> ProgressBar {
    let spinner = ProgressBar::new_spinner();
        spinner.set_style(ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .tick_chars("⠙⠹⠸⠼⠴⠦⠧⠇⠏"));
    spinner.enable_steady_tick(100);
    return spinner;
}
