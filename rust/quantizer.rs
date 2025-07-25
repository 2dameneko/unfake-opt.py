use ndarray::{Array3, ArrayView3};
use numpy::{PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy)]
struct Box3D {
    r_min: usize,
    r_max: usize,
    g_min: usize,
    g_max: usize,
    b_min: usize,
    b_max: usize,
    volume: i64, // Actually stores variance during cutting
}

impl Box3D {
    fn new() -> Self {
        Box3D {
            r_min: 0,
            r_max: 0,
            g_min: 0,
            g_max: 0,
            b_min: 0,
            b_max: 0,
            volume: 0,
        }
    }
}

#[pyclass]
pub struct WuQuantizerRust {
    weights: Array3<i64>,
    moments_r: Array3<i64>,
    moments_g: Array3<i64>,
    moments_b: Array3<i64>,
    moments: Array3<f64>,
    max_colors: usize,
    significant_bits: u8,
    side_size: usize,
    max_side_index: usize,
    boxes: Vec<Box3D>,
}

#[pymethods]
impl WuQuantizerRust {
    #[new]
    fn new(max_colors: usize, significant_bits: u8) -> Self {
        let side_size = 1 << significant_bits;
        let max_side_index = side_size - 1;

        Self {
            weights: Array3::<i64>::zeros((side_size, side_size, side_size)),
            moments_r: Array3::<i64>::zeros((side_size, side_size, side_size)),
            moments_g: Array3::<i64>::zeros((side_size, side_size, side_size)),
            moments_b: Array3::<i64>::zeros((side_size, side_size, side_size)),
            moments: Array3::<f64>::zeros((side_size, side_size, side_size)),
            max_colors,
            significant_bits,
            side_size,
            max_side_index,
            boxes: vec![Box3D::new(); max_colors],
        }
    }

    fn quantize(
        &mut self,
        py: Python<'_>,
        pixels: PyReadonlyArray3<u8>,
    ) -> PyResult<(Py<PyArray3<u8>>, Vec<(u8, u8, u8)>)> {
        let img_array = pixels.as_array();

        // Reset histograms
        self.weights.fill(0);
        self.moments_r.fill(0);
        self.moments_g.fill(0);
        self.moments_b.fill(0);
        self.moments.fill(0.0);

        // Build histogram
        self.build_histogram(&img_array);

        // Compute cumulative moments
        self.compute_cumulative_moments();

        // Initialize boxes
        self.boxes = vec![Box3D::new(); self.max_colors];
        self.boxes[0].r_max = self.max_side_index;
        self.boxes[0].g_max = self.max_side_index;
        self.boxes[0].b_max = self.max_side_index;

        // Perform box cutting
        let n_boxes = self.build_color_boxes();

        // Generate palette from boxes
        let palette = self.generate_palette(n_boxes);

        // Map pixels to palette
        let output = self.map_pixels_to_palette(py, &img_array, &palette)?;

        Ok((output, palette))
    }
}

impl WuQuantizerRust {
    fn build_histogram(&mut self, pixels: &ArrayView3<u8>) {
        let (height, width, channels) = pixels.dim();
        let has_alpha = channels == 4;
        let shift = 8 - self.significant_bits;

        // Parallel histogram building
        let histogram_parts: Vec<_> = (0..height)
            .into_par_iter()
            .map(|y| {
                let mut local_weights =
                    Array3::<i64>::zeros((self.side_size, self.side_size, self.side_size));
                let mut local_moments_r =
                    Array3::<i64>::zeros((self.side_size, self.side_size, self.side_size));
                let mut local_moments_g =
                    Array3::<i64>::zeros((self.side_size, self.side_size, self.side_size));
                let mut local_moments_b =
                    Array3::<i64>::zeros((self.side_size, self.side_size, self.side_size));
                let mut local_moments =
                    Array3::<f64>::zeros((self.side_size, self.side_size, self.side_size));

                for x in 0..width {
                    if has_alpha && pixels[(y, x, 3)] < 128 {
                        continue; // Skip transparent pixels
                    }

                    let r = pixels[(y, x, 0)];
                    let g = pixels[(y, x, 1)];
                    let b = pixels[(y, x, 2)];

                    let ir = (r >> shift) as usize;
                    let ig = (g >> shift) as usize;
                    let ib = (b >> shift) as usize;

                    local_weights[(ir, ig, ib)] += 1;
                    local_moments_r[(ir, ig, ib)] += r as i64;
                    local_moments_g[(ir, ig, ib)] += g as i64;
                    local_moments_b[(ir, ig, ib)] += b as i64;
                    local_moments[(ir, ig, ib)] +=
                        (r as i64 * r as i64 + g as i64 * g as i64 + b as i64 * b as i64) as f64;
                }

                (
                    local_weights,
                    local_moments_r,
                    local_moments_g,
                    local_moments_b,
                    local_moments,
                )
            })
            .collect();

        // Merge histograms
        for (local_w, local_r, local_g, local_b, local_m) in histogram_parts {
            self.weights += &local_w;
            self.moments_r += &local_r;
            self.moments_g += &local_g;
            self.moments_b += &local_b;
            self.moments += &local_m;
        }
    }

    fn compute_cumulative_moments(&mut self) {
        let size = self.side_size;

        // Cumulative in each dimension
        for r in 0..size {
            for g in 0..size {
                for b in 1..size {
                    self.weights[(r, g, b)] += self.weights[(r, g, b - 1)];
                    self.moments_r[(r, g, b)] += self.moments_r[(r, g, b - 1)];
                    self.moments_g[(r, g, b)] += self.moments_g[(r, g, b - 1)];
                    self.moments_b[(r, g, b)] += self.moments_b[(r, g, b - 1)];
                    self.moments[(r, g, b)] += self.moments[(r, g, b - 1)];
                }
            }
        }

        for r in 0..size {
            for b in 0..size {
                for g in 1..size {
                    self.weights[(r, g, b)] += self.weights[(r, g - 1, b)];
                    self.moments_r[(r, g, b)] += self.moments_r[(r, g - 1, b)];
                    self.moments_g[(r, g, b)] += self.moments_g[(r, g - 1, b)];
                    self.moments_b[(r, g, b)] += self.moments_b[(r, g - 1, b)];
                    self.moments[(r, g, b)] += self.moments[(r, g - 1, b)];
                }
            }
        }

        for g in 0..size {
            for b in 0..size {
                for r in 1..size {
                    self.weights[(r, g, b)] += self.weights[(r - 1, g, b)];
                    self.moments_r[(r, g, b)] += self.moments_r[(r - 1, g, b)];
                    self.moments_g[(r, g, b)] += self.moments_g[(r - 1, g, b)];
                    self.moments_b[(r, g, b)] += self.moments_b[(r - 1, g, b)];
                    self.moments[(r, g, b)] += self.moments[(r - 1, g, b)];
                }
            }
        }
    }

    fn volume(&self, cube: &Box3D, moment: &Array3<i64>) -> i64 {
        let r_min = if cube.r_min > 0 { cube.r_min - 1 } else { 0 };
        let g_min = if cube.g_min > 0 { cube.g_min - 1 } else { 0 };
        let b_min = if cube.b_min > 0 { cube.b_min - 1 } else { 0 };

        let mut vol = moment[(cube.r_max, cube.g_max, cube.b_max)];

        if cube.g_min > 0 {
            vol -= moment[(cube.r_max, g_min, cube.b_max)];
        }
        if cube.b_min > 0 {
            vol -= moment[(cube.r_max, cube.g_max, b_min)];
        }
        if cube.g_min > 0 && cube.b_min > 0 {
            vol += moment[(cube.r_max, g_min, b_min)];
        }
        if cube.r_min > 0 {
            vol -= moment[(r_min, cube.g_max, cube.b_max)];
            if cube.g_min > 0 {
                vol += moment[(r_min, g_min, cube.b_max)];
            }
            if cube.b_min > 0 {
                vol += moment[(r_min, cube.g_max, b_min)];
            }
            if cube.g_min > 0 && cube.b_min > 0 {
                vol -= moment[(r_min, g_min, b_min)];
            }
        }

        vol
    }

    fn volume_float(&self, cube: &Box3D, moment: &Array3<f64>) -> f64 {
        let r_min = if cube.r_min > 0 { cube.r_min - 1 } else { 0 };
        let g_min = if cube.g_min > 0 { cube.g_min - 1 } else { 0 };
        let b_min = if cube.b_min > 0 { cube.b_min - 1 } else { 0 };

        let mut vol = moment[(cube.r_max, cube.g_max, cube.b_max)];

        if cube.g_min > 0 {
            vol -= moment[(cube.r_max, g_min, cube.b_max)];
        }
        if cube.b_min > 0 {
            vol -= moment[(cube.r_max, cube.g_max, b_min)];
        }
        if cube.g_min > 0 && cube.b_min > 0 {
            vol += moment[(cube.r_max, g_min, b_min)];
        }
        if cube.r_min > 0 {
            vol -= moment[(r_min, cube.g_max, cube.b_max)];
            if cube.g_min > 0 {
                vol += moment[(r_min, g_min, cube.b_max)];
            }
            if cube.b_min > 0 {
                vol += moment[(r_min, cube.g_max, b_min)];
            }
            if cube.g_min > 0 && cube.b_min > 0 {
                vol -= moment[(r_min, g_min, b_min)];
            }
        }

        vol
    }

    fn bottom(&self, cube: &Box3D, direction: usize, moment: &Array3<i64>) -> i64 {
        match direction {
            0 => {
                if cube.r_min > 0 {
                    -self.volume(
                        &Box3D {
                            r_max: cube.r_min - 1,
                            ..*cube
                        },
                        moment,
                    )
                } else {
                    0
                }
            }
            1 => {
                if cube.g_min > 0 {
                    -self.volume(
                        &Box3D {
                            g_max: cube.g_min - 1,
                            ..*cube
                        },
                        moment,
                    )
                } else {
                    0
                }
            }
            _ => {
                if cube.b_min > 0 {
                    -self.volume(
                        &Box3D {
                            b_max: cube.b_min - 1,
                            ..*cube
                        },
                        moment,
                    )
                } else {
                    0
                }
            }
        }
    }

    fn top(&self, cube: &Box3D, direction: usize, position: usize, moment: &Array3<i64>) -> i64 {
        match direction {
            0 => self.volume(
                &Box3D {
                    r_min: position,
                    ..*cube
                },
                moment,
            ),
            1 => self.volume(
                &Box3D {
                    g_min: position,
                    ..*cube
                },
                moment,
            ),
            _ => self.volume(
                &Box3D {
                    b_min: position,
                    ..*cube
                },
                moment,
            ),
        }
    }

    fn variance(&self, cube: &Box3D) -> f64 {
        let dr = self.volume(cube, &self.moments_r) as f64;
        let dg = self.volume(cube, &self.moments_g) as f64;
        let db = self.volume(cube, &self.moments_b) as f64;
        let dm = self.volume_float(cube, &self.moments);
        let dw = self.volume(cube, &self.weights) as f64;

        if dw == 0.0 {
            0.0
        } else {
            dm - (dr * dr + dg * dg + db * db) / dw
        }
    }

    fn maximize(
        &self,
        cube: &Box3D,
        direction: usize,
        first: usize,
        last: usize,
        whole_r: i64,
        whole_g: i64,
        whole_b: i64,
        whole_w: i64,
    ) -> (f64, isize) {
        let bottom_r = self.bottom(cube, direction, &self.moments_r);
        let bottom_g = self.bottom(cube, direction, &self.moments_g);
        let bottom_b = self.bottom(cube, direction, &self.moments_b);
        let bottom_w = self.bottom(cube, direction, &self.weights);

        let mut max_variance = 0.0;
        let mut cut_position = -1;

        for i in first..last {
            let half_r = bottom_r + self.top(cube, direction, i, &self.moments_r);
            let half_g = bottom_g + self.top(cube, direction, i, &self.moments_g);
            let half_b = bottom_b + self.top(cube, direction, i, &self.moments_b);
            let half_w = bottom_w + self.top(cube, direction, i, &self.weights);

            if half_w == 0 {
                continue;
            }

            let mut temp = (half_r as f64 * half_r as f64
                + half_g as f64 * half_g as f64
                + half_b as f64 * half_b as f64)
                / half_w as f64;

            let half_r = whole_r - half_r;
            let half_g = whole_g - half_g;
            let half_b = whole_b - half_b;
            let half_w = whole_w - half_w;

            if half_w == 0 {
                continue;
            }

            temp += (half_r as f64 * half_r as f64
                + half_g as f64 * half_g as f64
                + half_b as f64 * half_b as f64)
                / half_w as f64;

            if temp > max_variance {
                max_variance = temp;
                cut_position = i as isize;
            }
        }

        (max_variance, cut_position)
    }

    fn cut(&mut self, set1: usize, set2: usize) -> bool {
        let whole_r = self.volume(&self.boxes[set1], &self.moments_r);
        let whole_g = self.volume(&self.boxes[set1], &self.moments_g);
        let whole_b = self.volume(&self.boxes[set1], &self.moments_b);
        let whole_w = self.volume(&self.boxes[set1], &self.weights);

        let (max_r, cut_r) = self.maximize(
            &self.boxes[set1],
            0,
            self.boxes[set1].r_min + 1,
            self.boxes[set1].r_max + 1,
            whole_r,
            whole_g,
            whole_b,
            whole_w,
        );
        let (max_g, cut_g) = self.maximize(
            &self.boxes[set1],
            1,
            self.boxes[set1].g_min + 1,
            self.boxes[set1].g_max + 1,
            whole_r,
            whole_g,
            whole_b,
            whole_w,
        );
        let (max_b, cut_b) = self.maximize(
            &self.boxes[set1],
            2,
            self.boxes[set1].b_min + 1,
            self.boxes[set1].b_max + 1,
            whole_r,
            whole_g,
            whole_b,
            whole_w,
        );

        let direction;
        let cut_position;

        if max_r >= max_g && max_r >= max_b {
            direction = 0;
            if cut_r < 0 {
                return false;
            }
            cut_position = cut_r as usize;
        } else if max_g >= max_b {
            direction = 1;
            cut_position = cut_g as usize;
        } else {
            direction = 2;
            cut_position = cut_b as usize;
        }

        // Copy box
        self.boxes[set2] = self.boxes[set1];

        // Do the cut
        match direction {
            0 => {
                self.boxes[set1].r_max = cut_position - 1;
                self.boxes[set2].r_min = cut_position;
            }
            1 => {
                self.boxes[set1].g_max = cut_position - 1;
                self.boxes[set2].g_min = cut_position;
            }
            _ => {
                self.boxes[set1].b_max = cut_position - 1;
                self.boxes[set2].b_min = cut_position;
            }
        }

        // Update volumes (actually variances for next cut selection)
        self.boxes[set1].volume = self.variance(&self.boxes[set1]) as i64;
        self.boxes[set2].volume = self.variance(&self.boxes[set2]) as i64;

        true
    }

    fn build_color_boxes(&mut self) -> usize {
        let mut n_boxes = 1;
        let mut next_box = 0;

        for i in 1..self.max_colors {
            if self.cut(next_box, i) {
                n_boxes += 1;

                if self.boxes[next_box].volume <= 0 {
                    self.boxes[next_box].volume = 0;
                }
                if self.boxes[i].volume <= 0 {
                    self.boxes[i].volume = 0;
                }
            } else {
                self.boxes[next_box].volume = 0;
            }

            // Find box with largest variance for next cut
            next_box = 0;
            let mut temp = self.boxes[0].volume;

            for j in 1..=i {
                if self.boxes[j].volume > temp {
                    temp = self.boxes[j].volume;
                    next_box = j;
                }
            }

            if temp <= 0 {
                n_boxes = i + 1;
                break;
            }
        }

        n_boxes
    }

    fn generate_palette(&self, n_boxes: usize) -> Vec<(u8, u8, u8)> {
        let mut palette = Vec::new();

        for i in 0..n_boxes {
            let weight = self.volume(&self.boxes[i], &self.weights);
            if weight > 0 {
                let r = (self.volume(&self.boxes[i], &self.moments_r) / weight) as u8;
                let g = (self.volume(&self.boxes[i], &self.moments_g) / weight) as u8;
                let b = (self.volume(&self.boxes[i], &self.moments_b) / weight) as u8;
                palette.push((r, g, b));
            }
        }

        palette
    }

    fn map_pixels_to_palette(
        &self,
        py: Python<'_>,
        pixels: &ArrayView3<u8>,
        palette: &[(u8, u8, u8)],
    ) -> PyResult<Py<PyArray3<u8>>> {
        let (height, width, channels) = pixels.dim();
        let has_alpha = channels == 4;

        // Create output array
        let output_array = unsafe { PyArray3::<u8>::new(py, [height, width, channels], false) };

        // Create a thread-safe wrapper for collecting results
        let results = Mutex::new(Vec::with_capacity(height * width));

        // Process pixels in parallel chunks
        let chunk_size = 64; // Process in 64x64 chunks for cache efficiency

        (0..height)
            .into_par_iter()
            .step_by(chunk_size)
            .for_each(|y_start| {
                let y_end = (y_start + chunk_size).min(height);
                let mut chunk_results = Vec::with_capacity((y_end - y_start) * width);

                for x_start in (0..width).step_by(chunk_size) {
                    let x_end = (x_start + chunk_size).min(width);

                    // Process chunk
                    for y in y_start..y_end {
                        for x in x_start..x_end {
                            if has_alpha && pixels[(y, x, 3)] < 128 {
                                // Transparent pixel
                                chunk_results.push((y, x, 0, 0, 0, 0));
                            } else {
                                // Find nearest palette color
                                let pixel_r = pixels[(y, x, 0)] as i32;
                                let pixel_g = pixels[(y, x, 1)] as i32;
                                let pixel_b = pixels[(y, x, 2)] as i32;

                                let mut min_dist = i32::MAX;
                                let mut best_color = &palette[0];

                                for color in palette {
                                    let dr = pixel_r - color.0 as i32;
                                    let dg = pixel_g - color.1 as i32;
                                    let db = pixel_b - color.2 as i32;
                                    let dist = dr * dr + dg * dg + db * db;

                                    if dist < min_dist {
                                        min_dist = dist;
                                        best_color = color;
                                    }
                                }

                                let a = if has_alpha { 255 } else { 0 };
                                chunk_results.push((
                                    y,
                                    x,
                                    best_color.0,
                                    best_color.1,
                                    best_color.2,
                                    a,
                                ));
                            }
                        }
                    }
                }

                results.lock().unwrap().extend(chunk_results);
            });

        // Write results to output array
        let results = results.into_inner().unwrap();
        for (y, x, r, g, b, a) in results {
            unsafe {
                *output_array.uget_mut([y, x, 0]) = r;
                *output_array.uget_mut([y, x, 1]) = g;
                *output_array.uget_mut([y, x, 2]) = b;
                if has_alpha {
                    *output_array.uget_mut([y, x, 3]) = a;
                }
            }
        }

        Ok(output_array.into())
    }
}
