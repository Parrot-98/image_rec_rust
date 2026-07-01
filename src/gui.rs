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
                    let mut min_row = 28;
                    let mut max_row = 0;
                    let mut min_col = 28;
                    let mut max_col = 0;
                    let mut has_pixels = false;

                    for row in 0..28 {
                        for col in 0..28 {
                            if self.grid[row][col] > 0.0 {
                                if row < min_row { min_row = row; }
                                if row > max_row { max_row = row; }
                                if col < min_col { min_col = col; }
                                if col > max_col { max_col = col; }
                                has_pixels = true;
                            }
                        }
                    }

                    let mut final_grid = [[0.0; 28]; 28];

                    if has_pixels {
                        let box_height = (max_row - min_row + 1) as f32;
                        let box_width = (max_col - min_col + 1) as f32;

                        let target_dim = 20.0;
                        let scale = (target_dim / box_height.max(box_width)).min(1.0);

                        let mut scaled_grid = [[0.0; 28]; 28];

                        // scale down the drwing
                        for row in min_row..=max_row {
                            for col in min_col..=max_col {
                                let val = self.grid[row][col];
                                if val > 0.0 {
                                    // Map old coordinates relative to bounding box center, then scale
                                    let rel_row = (row as f32 - min_row as f32) - (box_height / 2.0);
                                    let rel_col = (col as f32 - min_col as f32) - (box_width / 2.0);

                                    // Project onto the temporary grid centered at index 14
                                    let target_row = (14.0 + rel_row * scale).round() as i32;
                                    let target_col = (14.0 + rel_col * scale).round() as i32;

                                    if target_row >= 0 && target_row < 28 && target_col >= 0 && target_col < 28 {
                                        scaled_grid[target_row as usize][target_col as usize] = val;
                                    }
                                }
                            }
                        }

                        // recenter
                        let mut total_mass = 0.0;
                        let mut sum_row = 0.0;
                        let mut sum_col = 0.0;

                        for row in 0..28 {
                            for col in 0..28 {
                                let val = scaled_grid[row][col];
                                if val > 0.0 {
                                    total_mass += val;
                                    sum_row += row as f32 * val;
                                    sum_col += col as f32 * val;
                                }
                            }
                        }

                        if total_mass > 0.0 {
                            let center_row = sum_row / total_mass;
                            let center_col = sum_col / total_mass;

                            let shift_row = (13.5 - center_row).round() as i32;
                            let shift_col = (13.5 - center_col).round() as i32;

                            for row in 0..28 {
                                for col in 0..28 {
                                    let val = scaled_grid[row][col];
                                    if val > 0.0 {
                                        let t_row = row as i32 + shift_row;
                                        let t_col = col as i32 + shift_col;

                                        if t_row >= 0 && t_row < 28 && t_col >= 0 && t_col < 28 {
                                            final_grid[t_row as usize][t_col as usize] = val;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        final_grid = self.grid;
                    }

                    // flaten
                    let mut flattened = Vec::with_capacity(784);
                    for row in 0..28 {
                        for col in 0..28 {
                            flattened.push(final_grid[row][col]);
                        }
                    }

                    let input_matrix = Array2::from_shape_vec((1, 784), flattened).unwrap();

                    let hidden1 = (input_matrix.dot(&self.layer1.weights) + &self.layer1.bias)
                        .mapv(|x| if x > 0.0 { x } else { x * 0.01 });
                    let hidden2 = (hidden1.dot(&self.layer2.weights) + &self.layer2.bias)
                        .mapv(|x| if x > 0.0 { x } else { x * 0.01 });
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