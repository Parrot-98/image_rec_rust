mod load_data;
mod layers;
mod math;
mod training;
mod backpropagation;

fn main() {
    let (trn_img_normalized, trn_lbl, _tst_img_normalized, _tst_lbl) = load_data::load_data();

    let training_matrix = layers::Layer::get_data(trn_img_normalized);

    let mut first_hidden_layer = layers::Layer::new(784, 128);
    let mut second_hidden_layer = layers::Layer::new(128, 128);
    let mut third_hidden_layer = layers::Layer::new(128, 10);

    for epoch in 0..training_matrix.nrows() {
        let cost = training::train(
            &training_matrix.view(),
            &trn_lbl,
            &mut first_hidden_layer,
            &mut second_hidden_layer,
            &mut third_hidden_layer,
            epoch
        );

        if epoch % 100 == 0 {
            println!("cost: {}", cost);
        }
    }
}