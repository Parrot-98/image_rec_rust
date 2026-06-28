use ndarray::{Array1, ArrayBase, ArrayView2};
use crate::layers::Layer; 
use crate::math;

pub fn train(
    training_matrix: &ArrayView2<f32>,
    labels: &Vec<u8>,
    layer1: &Layer,
    layer2: &Layer,
    layer3: &Layer,
) -> Array1<f32> {
    for i in 0..training_matrix.nrows() {
        let input_layer = training_matrix.row(i);
        let _correct_label = labels[i];

        // calc 1
        let first_hidden_layer_non_bias = math::multiply(&input_layer, &layer1.weights.view());
        let first_hidden_layer = math::add(&first_hidden_layer_non_bias.view(), &layer1.bias.view());

        // calc 2
        let second_hidden_layer_non_bias = math::multiply(&first_hidden_layer.view(), &layer2.weights.view());
        let second_hidden_layer = math::add(&second_hidden_layer_non_bias.view(), &layer2.bias.view());

        // calc 3
        let output_non_bias = math::multiply(&second_hidden_layer.view(), &layer3.weights.view());
        let output = math::add(&output_non_bias.view(), &layer3.bias.view());

        return output;

    }

    Array1::zeros(10) // in case the top part fails(requierd)
}