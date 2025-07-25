use ndarray::{Array3, ArrayView3};
use rayon::prelude::*;
use std::collections::HashMap;

/// Downscale image using dominant color method
pub fn downscale_dominant(image: &ArrayView3<u8>, scale: usize, threshold: f32) -> Array3<u8> {
    let (height, width, channels) = image.dim();
    let target_h = height / scale;
    let target_w = width / scale;
    let has_alpha = channels == 4;

    // Collect results in parallel
    let results: Vec<_> = (0..target_h)
        .into_par_iter()
        .flat_map(|ty| {
            (0..target_w)
                .into_par_iter()
                .map(move |tx| {
                    let mut color_counts = HashMap::with_capacity(scale * scale);
                    let mut alpha_values = Vec::with_capacity(scale * scale);
                    let mut total_opaque = 0u32;

                    // Scan the scale x scale block
                    let y_start = ty * scale;
                    let x_start = tx * scale;
                    let y_end = ((ty + 1) * scale).min(height);
                    let x_end = ((tx + 1) * scale).min(width);

                    for y in y_start..y_end {
                        for x in x_start..x_end {
                            if has_alpha {
                                let alpha = image[(y, x, 3)];
                                alpha_values.push(alpha);

                                if alpha >= 128 {
                                    // Pack RGB into 24-bit integer (same as JavaScript)
                                    let r = image[(y, x, 0)] as u32;
                                    let g = image[(y, x, 1)] as u32;
                                    let b = image[(y, x, 2)] as u32;
                                    let color_key = (r << 16) | (g << 8) | b;

                                    *color_counts.entry(color_key).or_insert(0) += 1;
                                    total_opaque += 1;
                                }
                            } else {
                                let r = image[(y, x, 0)] as u32;
                                let g = image[(y, x, 1)] as u32;
                                let b = image[(y, x, 2)] as u32;
                                let color_key = (r << 16) | (g << 8) | b;

                                *color_counts.entry(color_key).or_insert(0) += 1;
                                total_opaque += 1;
                            }
                        }
                    }

                    // Calculate output pixel
                    let mut pixel = vec![0u8; channels];

                    if has_alpha && !alpha_values.is_empty() {
                        // Calculate median alpha efficiently
                        alpha_values.sort_unstable();
                        let median_alpha = alpha_values[alpha_values.len() / 2];
                        let final_alpha = if median_alpha >= 128 { 255 } else { 0 };

                        pixel[3] = final_alpha;

                        if final_alpha == 0 {
                            // Transparent pixel - set to black
                            pixel[0] = 0;
                            pixel[1] = 0;
                            pixel[2] = 0;
                            return (ty, tx, pixel);
                        }
                    }

                    if total_opaque > 0 {
                        // Find dominant color
                        let (dominant_color, max_count) = color_counts
                            .iter()
                            .max_by_key(|(_, &count)| count)
                            .map(|(&color, &count)| (color, count))
                            .unwrap_or((0, 0));

                        let threshold_count = (total_opaque as f32 * threshold) as u32;

                        if max_count >= threshold_count {
                            // Use dominant color
                            pixel[0] = ((dominant_color >> 16) & 0xFF) as u8;
                            pixel[1] = ((dominant_color >> 8) & 0xFF) as u8;
                            pixel[2] = (dominant_color & 0xFF) as u8;
                        } else {
                            // Calculate mean color
                            let mut r_sum = 0u64;
                            let mut g_sum = 0u64;
                            let mut b_sum = 0u64;

                            for (&color, &count) in &color_counts {
                                let count = count as u64;
                                r_sum += ((color >> 16) & 0xFF) as u64 * count;
                                g_sum += ((color >> 8) & 0xFF) as u64 * count;
                                b_sum += (color & 0xFF) as u64 * count;
                            }

                            pixel[0] = (r_sum / total_opaque as u64) as u8;
                            pixel[1] = (g_sum / total_opaque as u64) as u8;
                            pixel[2] = (b_sum / total_opaque as u64) as u8;
                        }

                        if !has_alpha {
                            pixel[3] = 255;
                        }
                    } else {
                        // No opaque pixels - set to black
                        pixel[0] = 0;
                        pixel[1] = 0;
                        pixel[2] = 0;
                        if !has_alpha {
                            pixel[3] = 255;
                        }
                    }

                    (ty, tx, pixel)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Build output array from results
    let mut output = Array3::<u8>::zeros((target_h, target_w, channels));
    for (ty, tx, pixel) in results {
        for (c, &value) in pixel.iter().enumerate() {
            output[(ty, tx, c)] = value;
        }
    }

    output
}
