use ndarray::{Array1, ArrayView1, ArrayView2};
use crate::math;

pub fn train(
    training_matrix: &ArrayView2<f32>,
    weights: &ArrayView2<f32>,
    weights_two: &ArrayView2<f32>,
    weights_three: &ArrayView2<f32>,
    bias: &ArrayView1<f32>,
    bias_two: &ArrayView1<f32>,
    bias_three: &ArrayView1<f32>,
) {
    
    for i in 0..training_matrix.len() {
        //calc 1
        let input_layer = training_matrix.row(i);

        let weights_input_layer = weights;
        let first_hidden_layer_non_bias = math::multiply(&input_layer, weights_input_layer);
        let first_hidden_layer = math::add(&first_hidden_layer_non_bias.view(), &bias);

        // calc 2
        let weights_hidden_layer_one = weights_two;
        let second_hidden_layer_non_bias = math::multiply(&first_hidden_layer.view(), &weights_hidden_layer_one);
        let second_hidden_layer = math::add(&second_hidden_layer_non_bias.view(), &bias_two);

        //calc 3
        let weights_hidden_layer_two = weights_three;
        let third_hidden_layer_non_bias = math::multiply(&second_hidden_layer.view(), &weights_hidden_layer_two);
        let output_layer = math::add(&third_hidden_layer_non_bias.view(), &bias_three);

        
    }

}