use ndarray::{Array1, Array2, ArrayView2, Axis};
use crate::math;

pub fn backpropagation(
    target: &ArrayView2<f32>,
    output: &ArrayView2<f32>,
    layer3_weights: &ArrayView2<f32>,
    layer2_weights: &ArrayView2<f32>,
    input_layer_three: &ArrayView2<f32>,
    input_layer_two: &ArrayView2<f32>,
    input_layer_one: &ArrayView2<f32>,
) -> (
    Array2<f32>, Array1<f32>,
    Array2<f32>, Array1<f32>,
    Array2<f32>, Array1<f32>,
) {
    let (layer_three_weight_gradient, layer_three_bias_gradient, delta_layer_three);
    let (layer_two_weight_gradient, layer_two_bias_gradient, delta_layer_two);
    let (layer_one_weight_gradient, layer_one_bias_gradient);

    let batch_size = output.nrows() as f32;

    {
        let delta_part_one = math::subtract(&output, &target);
        let relu_derivative = output.mapv(|x| if x > 0.0 { 1.0 } else { 1.0 });
        delta_layer_three = math::multiply_same(&delta_part_one.view(), &relu_derivative.view());

        layer_three_bias_gradient = delta_layer_three.sum_axis(Axis(0)) / batch_size;
        layer_three_weight_gradient = input_layer_three.t().dot(&delta_layer_three) / batch_size;
    }

    {
        let delta_passed_back = delta_layer_three.dot(&layer3_weights.t());
        let relu_derivative = input_layer_three.mapv(|x| if x > 0.0 { 1.0 } else { 1.0 });
        delta_layer_two = math::multiply_same(&delta_passed_back.view(), &relu_derivative.view());

        layer_two_bias_gradient = delta_layer_two.sum_axis(Axis(0)) / batch_size;
        layer_two_weight_gradient = input_layer_two.t().dot(&delta_layer_two) / batch_size;
    }

    {
        let delta_passed_back = delta_layer_two.dot(&layer2_weights.t());
        let relu_derivative = input_layer_two.mapv(|x| if x > 0.0 { 1.0 } else { 1.0 });
        let delta_layer_one = math::multiply_same(&delta_passed_back.view(), &relu_derivative.view());

        layer_one_bias_gradient = delta_layer_one.sum_axis(Axis(0)) / batch_size;
        layer_one_weight_gradient = input_layer_one.t().dot(&delta_layer_one) / batch_size;
    }

    (
        layer_three_weight_gradient, layer_three_bias_gradient,
        layer_two_weight_gradient, layer_two_bias_gradient,
        layer_one_weight_gradient, layer_one_bias_gradient,
    )
}