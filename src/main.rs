mod load_data;
mod layers;
mod math;
mod training;

fn main() {
    let (trn_img_normalized, trn_lbl, tst_img_normalized, tst_lbl) = load_data::load_data();

    let training_matrix = layers::Layer::get_data(trn_img_normalized, trn_lbl);
    println!("Data matrix shape: {:?}", training_matrix.shape());
    println!("{:?}", training_matrix.row(0));

    let x = layers::Layer::new(784, 128);

    let y = math::multiply(&training_matrix.row(0), &x.weights.view());
    println!("{:?}", y)

}