mod load_data;
mod layers;
mod math;
mod training;
mod backpropagation;

fn main() {
    let (trn_img_normalized, trn_lbl, _tst_img_normalized, _tst_lbl) = load_data::load_data();

    let training_matrix = layers::Layer::get_data(trn_img_normalized);

    let first_hidden_layer = layers::Layer::new(784, 128);
    let second_hidden_layer = layers::Layer::new(128, 128);
    let third_hidden_layer = layers::Layer::new(128, 10);




    let training = training::train(&training_matrix.view(), &trn_lbl, &first_hidden_layer, &second_hidden_layer, &third_hidden_layer);

    println!("{:?}", training);

}