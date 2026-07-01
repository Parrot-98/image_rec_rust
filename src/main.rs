mod load_data;
mod layers;
mod math;
mod training;
mod backpropagation;
mod init;
mod gui;

use std::io::{self, Write};
use std::path::Path;

fn main() {
    println!("Choose an execution mode:");
    println!(" (T) - run training");
    println!(" (E) - run tests");
    println!(" (G) - draw numbers");
    print!("> ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read line");
    let choice = input.trim().to_lowercase();
    let file_path = "weights.json";

    if choice == "t" || choice == "train" {
        init::training_init();
    } else if choice == "e" || choice == "test" {
        if Path::new(file_path).exists() {
            init::testing_init();
        } else {
            println!("Error: '{}' not found! Train your model first.", file_path);
        }
    } else if choice == "g" || choice == "draw" || choice == "gui" {
        gui::run_drawing_window();
    } else {
        println!("Invalid choice selected.");
    }
}