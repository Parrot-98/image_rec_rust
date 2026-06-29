use ndarray::{Array1, Array2, ArrayView1, Axis}; // Make sure to import Axis!
use crate::math;

pub fn backpropagation(
    target: &ArrayView1<f32>,
    output: &ArrayView1<f32>,
    input: &ArrayView1<f32>
) -> (Array2<f32>, Array1<f32>) {

    let ones_for_delta = Array1::ones(10);
    let delta_part_one = math::subtract(&output, &target);
    let delta_part_two = math::subtract(&ones_for_delta.view(), &output);
    let delta_part_three = math::multiply_same(&output, &delta_part_two.view());
    let delta = math::multiply_same(&delta_part_one.view(), &delta_part_three.view());

    let input_matrix = input.insert_axis(Axis(1));
    let bias_gradient = delta.clone(); 

    let delta_matrix = delta.insert_axis(Axis(0));
    let weight_gradient = input_matrix.dot(&delta_matrix);
    (weight_gradient, bias_gradient.clone())
}