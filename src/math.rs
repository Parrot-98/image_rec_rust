use ndarray::{Array1, ArrayView1, ArrayView2};
use std::process;


pub fn multiply(data: &ArrayView1<f32>, data2: &ArrayView2<f32>) -> Array1<f32> {
    let result = data.dot(data2);
    result
}

pub fn multiply_same(data: &ArrayView1<f32>, data2: &ArrayView1<f32>) -> Array1<f32> {
    let result = data * data2;
    result
}
// has tho be the same lenght
pub fn add(data: &ArrayView1<f32>, data2: &ArrayView1<f32>) -> Array1<f32>{
    if data.len() != data2.len() {
        eprintln!("the data is not the same lenght {} , {}", data, data2);
        process::exit(1);
    } else {
        let result = data + data2;
        result
    }
}
