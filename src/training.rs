use ndarray::{Array2, ArrayView2, Axis};
use crate::layers::Layer;
use crate::{backpropagation, math};

pub fn train(
    batch_inputs: &ArrayView2<f32>,
    batch_labels: &[u8],
    layer1: &mut Layer,
    layer2: &mut Layer,
    layer3: &mut Layer,
    is_training: bool,
) -> (f32, f32) {
    let batch_size = batch_inputs.nrows();
    let mut target = Array2::zeros((batch_size, 10));
    for idx in 0..batch_size {
        target[[idx, batch_labels[idx] as usize]] = 1.0;
    }
    
    // forwand pass
    let first_hidden_layer_non_bias = math::multiply(&batch_inputs.view(), &layer1.weights.view());
    let first_hidden_layer_non_relu = math::add(&first_hidden_layer_non_bias.view(), &layer1.bias.view());
    let first_hidden_layer = math::relu(&first_hidden_layer_non_relu.view());

    let second_hidden_layer_non_bias = math::multiply(&first_hidden_layer.view(), &layer2.weights.view());
    let second_hidden_layer_non_relu = math::add(&second_hidden_layer_non_bias.view(), &layer2.bias.view());
    let second_hidden_layer = math::relu(&second_hidden_layer_non_relu.view());

    let output_non_bias = math::multiply(&second_hidden_layer.view(), &layer3.weights.view());
    let output_non_relu = math::add(&output_non_bias.view(), &layer3.bias.view());
    let output = math::relu(&output_non_relu.view());

    let cost = math::cost(&output.view(), &target.view());

    // pridiction
    let predicted_classes = output.map_axis(Axis(1), |row| {
        let mut max_idx = 0;
        let mut max_val = row[0];
        for (idx, &val) in row.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }
        max_idx
    });

    let mut correct_predictions = 0;
    for idx in 0..batch_size {
        if predicted_classes[idx] == batch_labels[idx] as usize {
            correct_predictions += 1;
        }
    }
    let accuracy = (correct_predictions as f32 / batch_size as f32) * 100.0;

    if is_training {
        let (
            layer_three_weight_gradient, layer_three_bias_gradient,
            layer_two_weight_gradient, layer_two_bias_gradient,
            layer_one_weight_gradient, layer_one_bias_gradient,
        ) = backpropagation::backpropagation(
            &target.view(),
            &output.view(),
            &layer3.weights.view(),
            &layer2.weights.view(),
            &second_hidden_layer.view(),
            &first_hidden_layer.view(),
            &batch_inputs.view(),
        );

        let learning_rate = 0.01; // good numba

        // update
        layer3.weights.scaled_add(-learning_rate, &layer_three_weight_gradient);
        layer3.bias.scaled_add(-learning_rate, &layer_three_bias_gradient);

        layer2.weights.scaled_add(-learning_rate, &layer_two_weight_gradient);
        layer2.bias.scaled_add(-learning_rate, &layer_two_bias_gradient);

        layer1.weights.scaled_add(-learning_rate, &layer_one_weight_gradient);
        layer1.bias.scaled_add(-learning_rate, &layer_one_bias_gradient);
    }
    (cost, accuracy)
}