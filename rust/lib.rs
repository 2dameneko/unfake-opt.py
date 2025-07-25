use numpy::{PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Mutex;

mod quantizer;
mod utils;

use quantizer::WuQuantizerRust;
use utils::downscale_dominant;

/// Calculate GCD of two numbers
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Calculate GCD of a vector of numbers
fn gcd_of_vec(numbers: &[u32]) -> u32 {
    if numbers.is_empty() {
        return 1;
    }

    let mut result = numbers[0];
    for &num in &numbers[1..] {
        result = gcd(result, num);
        if result == 1 {
            return 1;
        }
    }
    result
}

/// Runs-based scale detection
#[pyfunction]
fn runs_based_detect(_py: Python<'_>, image: PyReadonlyArray3<u8>) -> PyResult<u32> {
    let img_array = image.as_array();
    let (height, width, channels) = img_array.dim();

    // Collect all run lengths in parallel
    let horizontal_runs: Vec<u32> = (0..height)
        .into_par_iter()
        .flat_map(|y| {
            let mut runs = Vec::new();
            let mut run_length = 1u32;

            for x in 1..width {
                let mut same = true;
                for c in 0..channels {
                    if img_array[(y, x, c)] != img_array[(y, x - 1, c)] {
                        same = false;
                        break;
                    }
                }

                if same {
                    run_length += 1;
                } else {
                    if run_length > 1 {
                        runs.push(run_length);
                    }
                    run_length = 1;
                }
            }

            if run_length > 1 {
                runs.push(run_length);
            }
            runs
        })
        .collect();

    let vertical_runs: Vec<u32> = (0..width)
        .into_par_iter()
        .flat_map(|x| {
            let mut runs = Vec::new();
            let mut run_length = 1u32;

            for y in 1..height {
                let mut same = true;
                for c in 0..channels {
                    if img_array[(y, x, c)] != img_array[(y - 1, x, c)] {
                        same = false;
                        break;
                    }
                }

                if same {
                    run_length += 1;
                } else {
                    if run_length > 1 {
                        runs.push(run_length);
                    }
                    run_length = 1;
                }
            }

            if run_length > 1 {
                runs.push(run_length);
            }
            runs
        })
        .collect();

    // Combine all runs
    let mut all_runs = horizontal_runs;
    all_runs.extend(vertical_runs);

    if all_runs.len() < 10 {
        return Ok(1);
    }

    // Calculate GCD of all run lengths
    let scale = gcd_of_vec(&all_runs);
    Ok(scale.max(1))
}

/// Map pixels to nearest palette colors
#[pyfunction]
fn map_pixels_to_palette(
    py: Python<'_>,
    pixels: PyReadonlyArray3<u8>,
    palette: Vec<(u8, u8, u8)>,
) -> PyResult<Py<PyArray3<u8>>> {
    let img_array = pixels.as_array();
    let (height, width, channels) = img_array.dim();
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
                        if has_alpha && img_array[(y, x, 3)] < 128 {
                            // Transparent pixel
                            chunk_results.push((y, x, 0, 0, 0, 0));
                        } else {
                            // Find nearest palette color
                            let pixel_r = img_array[(y, x, 0)] as i32;
                            let pixel_g = img_array[(y, x, 1)] as i32;
                            let pixel_b = img_array[(y, x, 2)] as i32;

                            let mut min_dist = i32::MAX;
                            let mut best_color = &palette[0];

                            for color in &palette {
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
                            chunk_results.push((y, x, best_color.0, best_color.1, best_color.2, a));
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

/// Downscale image using dominant color method
#[pyfunction]
fn downscale_dominant_color(
    _py: Python<'_>,
    image: PyReadonlyArray3<u8>,
    scale: usize,
    threshold: f32,
) -> PyResult<Py<PyArray3<u8>>> {
    let img_array = image.as_array();
    let result = downscale_dominant(&img_array, scale, threshold);

    // Convert ndarray::Array3 to PyArray3
    let (h, w, c) = result.dim();
    let py_array = unsafe { PyArray3::<u8>::new(_py, [h, w, c], false) };

    // Copy data
    for ((y, x, ch), &value) in result.indexed_iter() {
        unsafe {
            *py_array.uget_mut([y, x, ch]) = value;
        }
    }

    Ok(py_array.into())
}

/// Count unique opaque colors in an image
#[pyfunction]
fn count_unique_colors(_py: Python<'_>, image: PyReadonlyArray3<u8>) -> PyResult<usize> {
    let img_array = image.as_array();
    let (height, width, channels) = img_array.dim();
    let has_alpha = channels == 4;

    use std::collections::HashSet;
    let mut unique_colors = HashSet::new();

    for y in 0..height {
        for x in 0..width {
            // Skip transparent pixels
            if has_alpha && img_array[(y, x, 3)] < 128 {
                continue;
            }

            // Pack RGB into 24-bit integer
            let r = img_array[(y, x, 0)] as u32;
            let g = img_array[(y, x, 1)] as u32;
            let b = img_array[(y, x, 2)] as u32;
            let color_key = (r << 16) | (g << 8) | b;

            unique_colors.insert(color_key);
        }
    }

    Ok(unique_colors.len())
}

/// Finalize pixels by ensuring binary alpha and black transparent pixels
#[pyfunction]
fn finalize_pixels_rust(
    _py: Python<'_>,
    image: PyReadonlyArray3<u8>,
) -> PyResult<Py<PyArray3<u8>>> {
    let img_array = image.as_array();
    let (height, width, channels) = img_array.dim();

    if channels < 4 {
        // No alpha channel, return as-is
        let result = unsafe { PyArray3::<u8>::new(_py, [height, width, channels], false) };
        for ((y, x, c), &value) in img_array.indexed_iter() {
            unsafe {
                *result.uget_mut([y, x, c]) = value;
            }
        }
        return Ok(result.into());
    }

    // Has alpha channel - process it
    let result = unsafe { PyArray3::<u8>::new(_py, [height, width, channels], false) };

    for y in 0..height {
        for x in 0..width {
            let alpha = img_array[(y, x, 3)];

            if alpha < 128 {
                // Transparent - set to black with 0 alpha
                unsafe {
                    *result.uget_mut([y, x, 0]) = 0;
                    *result.uget_mut([y, x, 1]) = 0;
                    *result.uget_mut([y, x, 2]) = 0;
                    *result.uget_mut([y, x, 3]) = 0;
                }
            } else {
                // Opaque - copy color with 255 alpha
                unsafe {
                    *result.uget_mut([y, x, 0]) = img_array[(y, x, 0)];
                    *result.uget_mut([y, x, 1]) = img_array[(y, x, 1)];
                    *result.uget_mut([y, x, 2]) = img_array[(y, x, 2)];
                    *result.uget_mut([y, x, 3]) = 255;
                }
            }
        }
    }

    Ok(result.into())
}

/// Python module
#[pymodule]
fn unfake(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(runs_based_detect, m)?)?;
    m.add_function(wrap_pyfunction!(map_pixels_to_palette, m)?)?;
    m.add_function(wrap_pyfunction!(downscale_dominant_color, m)?)?;
    m.add_function(wrap_pyfunction!(count_unique_colors, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_pixels_rust, m)?)?;
    m.add_class::<WuQuantizerRust>()?;
    Ok(())
}
