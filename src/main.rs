mod load_data;
mod layers;
mod math;
mod training;
mod backpropagation;

use ndarray::s;

fn main() {
    let (trn_img_normalized, trn_lbl, _tst_img_normalized, _tst_lbl) = load_data::load_data();
    let training_matrix = layers::Layer::get_data(trn_img_normalized);

    let mut layer1 = layers::Layer::new(784, 128);
    let mut layer2 = layers::Layer::new(128, 128);
    let mut layer3 = layers::Layer::new(128, 10);

    let epochs = 5;
    let batch_size = 32;

    for epoch in 0..epochs {
        let mut total_cost = 0.0;
        let mut steps = 0;
        let num_samples = training_matrix.nrows();

        for i in (0..num_samples).step_by(batch_size) {
            // Guard against out-of-bounds slicing on the last batch
            let end = std::cmp::min(i + batch_size, num_samples);
            let current_batch_size = end - i;
            if current_batch_size < batch_size { break; } 

            let batch_inputs = training_matrix.slice(s![i..end, ..]);
            let batch_labels = &trn_lbl[i..end];

            let cost = training::train(
                &batch_inputs.to_owned().view(),
                batch_labels,
                &mut layer1,
                &mut layer2,
                &mut layer3,
            );

            total_cost += cost;
            steps += 1;
        }

        let average_cost = total_cost / steps as f32;
        println!("Epoch {} completed | Average Cost: {:.5}", epoch, average_cost);
    }
}