use mnist::{Mnist, MnistBuilder};

pub fn load_data() -> (Vec<f32>, Vec<u8>, Vec<f32>, Vec<u8>){
    let Mnist { 
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .test_set_length(10_000)
        .base_path("data/")
        .training_images_filename("train-images.idx3-ubyte")
        .training_labels_filename("train-labels.idx1-ubyte")
        .test_images_filename("t10k-images.idx3-ubyte")
        .test_labels_filename("t10k-labels.idx1-ubyte")
        .finalize();

    let trn_img_normalized: Vec<f32> = trn_img
        .iter()
        .map(|&pixel| pixel as f32 / 255.0)
        .collect();

    let tst_img_normalized: Vec<f32> = tst_img
        .iter()
        .map(|&pixel| pixel as f32 / 255.0)
        .collect();

    (trn_img_normalized, trn_lbl, tst_img_normalized, tst_lbl)


}