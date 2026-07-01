use crate::layers::Layer;
use eframe::egui;
use ndarray::Array2;
use std::fs::File;

pub struct DigitRecognizerApp {
    grid: [[f32; 28]; 28],
    prediction: Option<usize>,
    layer1: Layer,
    layer2: Layer,
    layer3: Layer,
}

impl DigitRecognizerApp {
    pub fn new() -> Self {
        let load_file = File::open("weights.json")
            .expect("Missing weights.json! Run training via the CLI first.");
        let (layer1, layer2, layer3): (Layer, Layer, Layer) =
            serde_json::from_reader(load_file).expect("Failed parsing weights");

        Self {
            grid: [[0.0; 28]; 28],
            prediction: None,
            layer1,
            layer2,
            layer3,
        }
    }
}

impl eframe::App for DigitRecognizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Draw a Digit (0-9)");

            ui.horizontal(|ui| {
                if ui.button("Clear Canvas").clicked() {
                    self.grid = [[0.0; 28]; 28];
                    self.prediction = None;
                }

                if ui.button("Predict Digit").clicked() {
                    let mut flattened = Vec::with_capacity(784);
                    for row in 0..28 {
                        for col in 0..28 {
                            flattened.push(self.grid[row][col]);
                        }
                    }

                    let input_matrix = Array2::from_shape_vec((1, 784), flattened).unwrap();

                    let hidden1 = (input_matrix.dot(&self.layer1.weights) + &self.layer1.bias)
                        .mapv(|x| if x > 0.0 { x } else { 0.0 });
                    let hidden2 = (hidden1.dot(&self.layer2.weights) + &self.layer2.bias)
                        .mapv(|x| if x > 0.0 { x } else { 0.0 });
                    let output = hidden2.dot(&self.layer3.weights) + &self.layer3.bias;

                    let mut max_val = f32::MIN;
                    let mut max_idx = 0;
                    for (idx, &val) in output.iter().enumerate() {
                        if val > max_val {
                            max_val = val;
                            max_idx = idx;
                        }
                    }
                    self.prediction = Some(max_idx);
                }
            });

            ui.add_space(10.0);

            let (rect, response) = ui.allocate_exact_size(
                egui::vec2(420.0, 420.0),
                egui::Sense::drag(),
            );

            ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);
            let cell_size = 420.0 / 28.0;

            if response.dragged() || response.clicked() {
                if let Some(pointer_pos) = response.interact_pointer_pos() {
                    let local_pos = pointer_pos - rect.min;
                    let brush_radius = 1.5; 

                    for row in 0..28 {
                        for col in 0..28 {
                            let cell_x = (col as f32 * cell_size) + (cell_size / 2.0);
                            let cell_y = (row as f32 * cell_size) + (cell_size / 2.0);

                            let dx = local_pos.x - cell_x;
                            let dy = local_pos.y - cell_y;
                            let distance = (dx * dx + dy * dy).sqrt() / cell_size;

                            if distance <= brush_radius {
                                let intensity = 1.0 - (distance / brush_radius);
                                self.grid[row][col] = (self.grid[row][col] + intensity * 0.85).min(1.0);
                            }
                        }
                    }
                }
            }

            // Draw grid cells onto the frame
            for row in 0..28 {
                for col in 0..28 {
                    let val = self.grid[row][col];
                    if val > 0.0 {
                        let cell_rect = egui::Rect::from_min_size(
                            rect.min + egui::vec2(col as f32 * cell_size, row as f32 * cell_size),
                            egui::vec2(cell_size, cell_size),
                        );
                        let gray = (val * 255.0) as u8;
                        ui.painter().rect_filled(cell_rect, 0.0, egui::Color32::from_rgb(gray, gray, gray));
                    }
                }
            }

            ui.add_space(20.0);
            if let Some(digit) = self.prediction {
                ui.label(egui::RichText::new(format!("Prediction: {}", digit)).size(32.0).strong().color(egui::Color32::GREEN));
            } else {
                ui.label(egui::RichText::new("Prediction: None").size(32.0).strong());
            }
        });
    }
}

pub fn run_drawing_window() {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([460.0, 560.0]),
        ..Default::default()
    };
    
    eframe::run_native(
        "digit recognizer",
        native_options,
        Box::new(|_cc| Box::new(DigitRecognizerApp::new())),
    ).unwrap();
}