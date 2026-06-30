use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

pub fn multiply(matrix: &ArrayView2<f32>, weights: &ArrayView2<f32>) -> Array2<f32> {
    matrix.dot(weights)
}

// must be the same lenght
pub fn add(matrix: &ArrayView2<f32>, bias: &ArrayView1<f32>) -> Array2<f32> {
    matrix + bias
}

pub fn relu(matrix: &ArrayView2<f32>) -> Array2<f32> {
    matrix.mapv(|x| if x > 0.0 { x } else { 0.0 })
}

pub fn cost(output: &ArrayView2<f32>, target: &ArrayView2<f32>) -> f32 {
    let diff = output - target;
    let squared = &diff * &diff;
    0.5 * squared.sum() / output.nrows() as f32
}

pub fn subtract(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> Array2<f32> {
    a - b
}

pub fn multiply_same(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> Array2<f32> {
    a * b
}