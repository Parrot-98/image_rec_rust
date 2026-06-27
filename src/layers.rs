use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt; 
use ndarray_rand::rand_distr::Normal;

enum Types {
    Input,
    Hidden,
    Output,
}

pub struct Layer {
    pub weights: Array2<f32>,
    pub bias: Array1<f32>, 
}

struct ActivationNodes {
    kind: Types,
    values: Vec<f32>,
}

impl Layer {
    pub fn new(input_nodes: usize, output_nodes: usize) -> Self {
        let distribution = Normal::new(0.0, 0.1)
            .expect("Failed to create random distribution");

        let weights = Array2::random((input_nodes, output_nodes), distribution);
        let bias = Array1::zeros(output_nodes);

        Layer { weights, bias }
    }


    pub fn get_data(data: Vec<f32>, ans: Vec<u8>) -> Array2<f32> {
        let total_pixels = data.len();
        let total_images = total_pixels / 784;

        Array2::from_shape_vec((total_images, 784), data)
            .expect("Not the correct data") // cahange the data form Vec<f32> to Array 2
    }
}