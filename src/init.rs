use crate::layers;
use crate::training;
use crate::load_data;

use ndarray::s;
use std::fs::File;

const WEIGHTS_PATH: &str = "weights.json";
const BATCH_SIZE: usize = 32;

pub fn training_init() {
    let (trn_img_normalized, trn_lbl, _, _) = load_data::load_data();
    let training_matrix = layers::Layer::get_data(trn_img_normalized);

    // Initialize fresh random layers
    let mut layer1 = layers::Layer::new(784, 128);
    let mut layer2 = layers::Layer::new(128, 128);
    let mut layer3 = layers::Layer::new(128, 10);

    let epochs = 450; // noice numba

    println!("Starting Training Loop...");
    for epoch in 0..epochs {
        let mut total_cost = 0.0;
        let mut total_accuracy = 0.0;
        let mut steps = 0;
        let num_samples = training_matrix.nrows();
        let num_batches = 60000 / BATCH_SIZE;

        for i in (0..num_samples).step_by(BATCH_SIZE) {
            let end = std::cmp::min(i + BATCH_SIZE, num_samples);
            let current_batch_size = end - i;
            if current_batch_size < BATCH_SIZE { break; }

            let batch_inputs = training_matrix.slice(s![i..end, ..]);
            let batch_labels = &trn_lbl[i..end];

            let (cost, accuracy) = training::train(
                &batch_inputs.to_owned().view(),
                batch_labels,
                &mut layer1,
                &mut layer2,
                &mut layer3,
                true
            );

            total_cost += cost;
            total_accuracy += accuracy / num_batches as f32;
            steps += 1;
        }

        let average_cost = total_cost / steps as f32;
        println!("Epoch {} , Average Cost: {:.5} , accuracy: {:.2}%", epoch, average_cost, total_accuracy);
    }

    println!("\nSaving at: {}", WEIGHTS_PATH);
    let save_file = File::create(WEIGHTS_PATH).expect("Failed to create weights file");
    serde_json::to_writer_pretty(save_file, &(&layer1, &layer2, &layer3))
        .expect("Failed to serialize layer data");
    println!("Stored");
}

pub fn testing_init() {
    let (_, _, tst_img_normalized, tst_lbl) = load_data::load_data();
    let testing_matrix = layers::Layer::get_data(tst_img_normalized);

    let load_file = File::open(WEIGHTS_PATH).expect("Missing weights.json! Run training first.");
    let (mut layer1, mut layer2, mut layer3): (layers::Layer, layers::Layer, layers::Layer) = 
        serde_json::from_reader(load_file).expect("Failed parsing weights");
    println!("Weights successfully loaded!");

    let mut total_test_cost = 0.0;
    let mut total_test_accuracy = 0.0;
    let mut test_steps = 0;

    let num_test_samples = testing_matrix.nrows();
    let num_test_batches = 10000 / BATCH_SIZE;

    for i in (0..num_test_samples).step_by(BATCH_SIZE) {
        let end = std::cmp::min(i + BATCH_SIZE, num_test_samples);
        if (end - i) < BATCH_SIZE { break; }

        let test_batch_inputs = testing_matrix.slice(s![i..end, ..]);
        let test_batch_labels = &tst_lbl[i..end];

        let (cost, accuracy) = training::train(
            &test_batch_inputs.to_owned().view(),
            test_batch_labels,
            &mut layer1,
            &mut layer2,
            &mut layer3,
            false
        );

        total_test_cost += cost;
        total_test_accuracy += accuracy / num_test_batches as f32;
        test_steps += 1;
    }

    let average_test_cost = total_test_cost / test_steps as f32;
    println!("Average Test Cost: {:.5}", average_test_cost);
    println!("Generalization Accuracy: {:.2}%", total_test_accuracy);
}