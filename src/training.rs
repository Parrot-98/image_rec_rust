use ndarray::{Array1, ArrayView2};
use crate::layers::Layer;
use crate::{backpropagation, math};

pub fn train(
    training_matrix: &ArrayView2<f32>,
    labels: &Vec<u8>,
    layer1: &mut Layer,
    layer2: &mut Layer,
    layer3: &mut Layer,
    i: usize,
) -> f32  {
    // forward pass
    let input_layer = training_matrix.row(i);
    let correct_label = labels[i];

    let mut target = Array1::zeros(10);
    target[correct_label as usize] = 1.0;// set the lable at the correct spot

    // calc 1
    let first_hidden_layer_non_bias = math::multiply(&input_layer, &layer1.weights.view());
    let first_hidden_layer_non_relu = math::add(&first_hidden_layer_non_bias.view(), &layer1.bias.view());
    let first_hidden_layer = math::relu(&first_hidden_layer_non_relu.view());

    // calc 2
    let second_hidden_layer_non_bias = math::multiply(&first_hidden_layer.view(), &layer2.weights.view());
    let second_hidden_layer_non_relu = math::add(&second_hidden_layer_non_bias.view(), &layer2.bias.view());
    let second_hidden_layer = math::relu(&second_hidden_layer_non_relu.view());

    // calc 3
    let output_non_bias = math::multiply(&second_hidden_layer.view(), &layer3.weights.view());
    let output = math::add(&output_non_bias.view(), &layer3.bias.view());

    // backward pass
    let cost = math::cost(&output.view(), &target.view());

    let (
        layer_three_weight_gradient, layer_three_bias_gradient,
        layer_two_weight_gradient, layer_two_bias_gradient,
        layer_one_weight_gradient, layer_one_bias_gradient,
    ) = backpropagation::backpropagation(&target.view(),
        &output.view(), 
        &layer3.weights.view(),
        &layer2.weights.view(),
        &second_hidden_layer.view(),
        &first_hidden_layer.view(),
        &input_layer.view(),
    );


    // update weights and bias
    let learning_rate = 0.001;

    layer3.weights.scaled_add(-learning_rate, &layer_three_weight_gradient);
    layer3.bias.scaled_add(-learning_rate, &layer_three_bias_gradient);

    layer2.weights.scaled_add(-learning_rate, &layer_two_weight_gradient);
    layer2.bias.scaled_add(-learning_rate, &layer_two_bias_gradient);

    layer1.weights.scaled_add(-learning_rate, &layer_one_weight_gradient);
    layer1.bias.scaled_add(-learning_rate, &layer_one_bias_gradient);

    return cost;

}