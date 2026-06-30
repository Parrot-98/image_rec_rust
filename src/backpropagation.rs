use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use crate::math;

pub fn backpropagation(
    target: &ArrayView1<f32>,
    output: &ArrayView1<f32>,
    layer3_weights: &ArrayView2<f32>,
    layer2_weights: &ArrayView2<f32>,
    input_layer_three: &ArrayView1<f32>,
    input_layer_two: &ArrayView1<f32>,
    input_layer_one: &ArrayView1<f32>,
) -> (
    Array2<f32>, Array1<f32>,
    Array2<f32>, Array1<f32>,
    Array2<f32>, Array1<f32>,
) {
    let (layer_three_weight_gradient, layer_three_bias_gradient, delta_layer_three);
    let (layer_two_weight_gradient, layer_two_bias_gradient, delta_layer_two);
    let (layer_one_weight_gradient, layer_one_bias_gradient);
    // layer 3
    {
        let ones_for_delta = Array1::ones(10);
        let delta_part_one = math::subtract(&output, &target);
        let delta_part_two = math::subtract(&ones_for_delta.view(), &output);
        let delta_part_three = math::multiply_same(&output, &delta_part_two.view());
        
        delta_layer_three = math::multiply_same(&delta_part_one.view(), &delta_part_three.view());

        let input_matrix = input_layer_three.insert_axis(Axis(1));
        let delta_matrix = delta_layer_three.clone().insert_axis(Axis(0));
        
        layer_three_bias_gradient = delta_layer_three.clone();
        layer_three_weight_gradient = input_matrix.dot(&delta_matrix);
    }
    // layer 2
    {
        let delta_passed_back = layer3_weights.dot(&delta_layer_three);
        let relu_derivative = input_layer_three.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        delta_layer_two = math::multiply_same(&delta_passed_back.view(), &relu_derivative.view());

        let input_matrix = input_layer_two.insert_axis(Axis(1));
        let delta_matrix = delta_layer_two.clone().insert_axis(Axis(0));

        layer_two_bias_gradient = delta_layer_two.clone();
        layer_two_weight_gradient = input_matrix.dot(&delta_matrix);
    }

    {
        let delta_passed_back = layer2_weights.dot(&delta_layer_two);
        let relu_derivative = input_layer_two.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        let delta_layer_one = math::multiply_same(&delta_passed_back.view(), &relu_derivative.view());

        let input_matrix = input_layer_one.insert_axis(Axis(1));
        let delta_matrix = delta_layer_one.clone().insert_axis(Axis(0));

        layer_one_bias_gradient = delta_layer_one.clone();
        layer_one_weight_gradient = input_matrix.dot(&delta_matrix);
    }
    // layer 1
    (
        layer_three_weight_gradient, layer_three_bias_gradient,
        layer_two_weight_gradient, layer_two_bias_gradient,
        layer_one_weight_gradient, layer_one_bias_gradient,
    )
}