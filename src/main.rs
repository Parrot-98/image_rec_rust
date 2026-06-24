use mnist::{Mnist, MnistBuilder};

fn main() {
    let Mnist { 
        trn_img,
        trn_lbl,
        tst_img: _tst_img,
        tst_lbl: _tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .base_path("data/")
        .training_images_filename("train-images.idx3-ubyte")
        .training_labels_filename("train-labels.idx1-ubyte")
        .test_images_filename("t10k-images.idx3-ubyte")
        .test_labels_filename("t10k-labels.idx1-ubyte")
        .finalize();

    let first_label = trn_lbl[1000];

    let first_image_pixels_raw = &trn_img[0..784]; 

    let first_image_pixels_normalized: Vec<f32> = first_image_pixels_raw
        .iter()
        .map(|&pixel| pixel as f32 / 255.0)
        .collect();

    println!("first image: {}", first_label);
    println!("pixel count: {}", first_image_pixels_normalized.len());
}