use ndarray::{Array3, ArrayView3};
use rayon::prelude::*;
use std::collections::HashMap;

use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Simple 2x2 matrix for covariance operations
#[derive(Clone, Copy, Debug)]
struct Matrix2x2 {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
}

impl Matrix2x2 {
    fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self { a, b, c, d }
    }

    fn multiply(&self, other: &Matrix2x2) -> Matrix2x2 {
        Matrix2x2::new(
            self.a * other.a + self.b * other.c,
            self.a * other.b + self.b * other.d,
            self.c * other.a + self.d * other.c,
            self.c * other.b + self.d * other.d,
        )
    }

    fn determinant(&self) -> f32 {
        self.a * self.d - self.b * self.c
    }

    fn inverse(&self) -> Option<Matrix2x2> {
        let det = self.determinant();
        if det.abs() < 1e-9 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Matrix2x2::new(
            self.d * inv_det,
            -self.b * inv_det,
            -self.c * inv_det,
            self.a * inv_det,
        ))
    }

    // Simple SVD for 2x2 symmetric positive definite matrices
    fn eigenvalues(&self) -> (f32, f32) {
        let trace = self.a + self.d;
        let det = self.determinant();
        let discriminant = (trace * trace - 4.0 * det).max(0.0).sqrt();
        let lambda1 = (trace + discriminant) * 0.5;
        let lambda2 = (trace - discriminant) * 0.5;
        (lambda1.max(0.0), lambda2.max(0.0))
    }

    fn clamp_eigenvalues(&self, min_val: f32, max_val: f32) -> Matrix2x2 {
        let (mut l1, mut l2) = self.eigenvalues();
        l1 = l1.clamp(min_val, max_val);
        l2 = l2.clamp(min_val, max_val);

        // For symmetric matrices, reconstruct using eigenvalues
        // This is a simplified reconstruction that preserves the matrix structure
        let scale1 = if self.a > 0.0 {
            (l1 / self.a.max(1e-9)).sqrt()
        } else {
            1.0
        };
        let scale2 = if self.d > 0.0 {
            (l2 / self.d.max(1e-9)).sqrt()
        } else {
            1.0
        };

        Matrix2x2::new(
            self.a * scale1,
            self.b * (scale1 + scale2) * 0.5,
            self.c * (scale1 + scale2) * 0.5,
            self.d * scale2,
        )
    }
}

/// Gaussian kernel for content-adaptive downscaling
#[derive(Clone)]
struct GaussianKernel {
    mu: [f32; 2],     // position (x, y)
    sigma: Matrix2x2, // covariance matrix
    nu: [f32; 3],     // color (L, a, b)
}

impl GaussianKernel {
    fn new(x: f32, y: f32, sx: f32, sy: f32) -> Self {
        Self {
            mu: [x, y],
            sigma: Matrix2x2::new(sx, 0.0, 0.0, sy),
            nu: [50.0, 0.0, 0.0], // neutral gray in LAB
        }
    }

    fn evaluate_weight(&self, x: f32, y: f32, inv_sigma: &Matrix2x2) -> f32 {
        let dx = x - self.mu[0];
        let dy = y - self.mu[1];
        let exponent = dx * dx * inv_sigma.a + 2.0 * dx * dy * inv_sigma.b + dy * dy * inv_sigma.d;
        (-0.5 * exponent).exp()
    }
}

/// Extremely fast content-adaptive downscaling using EM-C algorithm
pub fn content_adaptive_downscale_rust(
    lab_image: &ArrayView3<f32>,
    target_w: usize,
    target_h: usize,
    num_iterations: usize,
) -> Array3<f32> {
    let (h_in, w_in, _) = lab_image.dim();
    let h_out = target_h;
    let w_out = target_w;
    let rx = w_in as f32 / w_out as f32;
    let ry = h_in as f32 / h_out as f32;

    // Initialize kernels
    let mut kernels: Vec<GaussianKernel> = (0..h_out)
        .flat_map(|yk| {
            (0..w_out).map(move |xk| {
                let center_x = (xk as f32 + 0.5) * rx;
                let center_y = (yk as f32 + 0.5) * ry;
                let initial_sx = (rx / 3.0) * (rx / 3.0);
                let initial_sy = (ry / 3.0) * (ry / 3.0);
                GaussianKernel::new(center_x, center_y, initial_sx, initial_sy)
            })
        })
        .collect();

    let num_kernels = kernels.len();
    let search_radius = 2.0 * rx.max(ry);
    let search_radius_sq = search_radius * search_radius;

    // EM-C iterations
    for _ in 0..num_iterations {
        // Precompute inverse sigmas (always Some after clamping/initialization)
        let inv_sigmas: Vec<Matrix2x2> = kernels
            .par_iter()
            .map(|kernel| kernel.sigma.inverse().unwrap())
            .collect();

        // E-Step: Compute weights in parallel, using local pixel bounds per kernel for efficiency
        let mut all_weights: Vec<Vec<(usize, f32)>> = (0..num_kernels)
            .into_par_iter()
            .map(|k| {
                let kernel = &kernels[k];
                let inv_sigma = &inv_sigmas[k];

                let x_min = (kernel.mu[0] - search_radius).max(0.0) as usize;
                let x_max = (kernel.mu[0] + search_radius).min(w_in as f32 - 1.0) as usize + 1;
                let y_min = (kernel.mu[1] - search_radius).max(0.0) as usize;
                let y_max = (kernel.mu[1] + search_radius).min(h_in as f32 - 1.0) as usize + 1;

                let approx_capacity = (x_max - x_min) * (y_max - y_min);
                let mut weights = Vec::with_capacity(approx_capacity / 2); // Conservative estimate

                for y in y_min..y_max {
                    for x in x_min..x_max {
                        let fx = x as f32;
                        let fy = y as f32;
                        let dx = fx - kernel.mu[0];
                        let dy = fy - kernel.mu[1];
                        let dist_sq = dx * dx + dy * dy;
                        if dist_sq <= search_radius_sq {
                            let weight = kernel.evaluate_weight(fx, fy, inv_sigma);
                            if weight > 1e-5 {
                                let pixel_idx = y * w_in + x;
                                weights.push((pixel_idx, weight));
                            }
                        }
                    }
                }
                weights
            })
            .collect();

        // Normalize weights
        let mut gamma_sums = vec![1e-9f32; h_in * w_in];
        for k in 0..num_kernels {
            let w_sum: f32 = all_weights[k].iter().map(|&(_, w)| w).sum::<f32>() + 1e-9;
            for &mut (pixel_idx, ref mut weight) in all_weights[k].iter_mut() {
                *weight /= w_sum;
                gamma_sums[pixel_idx] += *weight;
            }
        }

        // M-Step: Update kernel parameters in parallel
        let new_kernels: Vec<GaussianKernel> = (0..num_kernels)
            .into_par_iter()
            .map(|k| {
                let weights = &all_weights[k];
                let mut w_sum = 1e-9f32;
                let mut new_mu = [0.0f32; 2];
                let mut new_nu = [0.0f32; 3];

                // Update position and color
                for &(pixel_idx, weight) in weights {
                    let y = pixel_idx / w_in;
                    let x = pixel_idx % w_in;
                    let gamma_k_i = weight / gamma_sums[pixel_idx];
                    w_sum += gamma_k_i;

                    new_mu[0] += gamma_k_i * x as f32;
                    new_mu[1] += gamma_k_i * y as f32;
                    new_nu[0] += gamma_k_i * lab_image[(y, x, 0)];
                    new_nu[1] += gamma_k_i * lab_image[(y, x, 1)];
                    new_nu[2] += gamma_k_i * lab_image[(y, x, 2)];
                }

                new_mu[0] /= w_sum;
                new_mu[1] /= w_sum;
                new_nu[0] /= w_sum;
                new_nu[1] /= w_sum;
                new_nu[2] /= w_sum;

                // Update covariance
                let mut cov_sum = [0.0f32; 4]; // [xx, xy, yx, yy]
                for &(pixel_idx, weight) in weights {
                    let y = pixel_idx / w_in;
                    let x = pixel_idx % w_in;
                    let gamma_k_i = weight / gamma_sums[pixel_idx];
                    let dx = x as f32 - new_mu[0];
                    let dy = y as f32 - new_mu[1];

                    cov_sum[0] += gamma_k_i * dx * dx;
                    cov_sum[1] += gamma_k_i * dx * dy;
                    cov_sum[3] += gamma_k_i * dy * dy;
                }

                cov_sum[0] /= w_sum;
                cov_sum[1] /= w_sum;
                cov_sum[2] = cov_sum[1]; // symmetric
                cov_sum[3] /= w_sum;

                let new_sigma = Matrix2x2::new(cov_sum[0], cov_sum[1], cov_sum[2], cov_sum[3]);

                GaussianKernel {
                    mu: new_mu,
                    sigma: new_sigma,
                    nu: new_nu,
                }
            })
            .collect();

        // C-Step: Clamp kernel sizes for sharpness
        kernels = new_kernels
            .into_iter()
            .map(|mut kernel| {
                kernel.sigma = kernel.sigma.clamp_eigenvalues(0.05, 0.1);
                kernel
            })
            .collect();
    }

    // Generate output image
    let mut output = Array3::<f32>::zeros((h_out, w_out, 3));
    for yk in 0..h_out {
        for xk in 0..w_out {
            let k_idx = yk * w_out + xk;
            if k_idx < kernels.len() {
                for c in 0..3 {
                    output[(yk, xk, c)] = kernels[k_idx].nu[c];
                }
            }
        }
    }

    output
}

/// Downscale image using mode (most frequent color) method
pub fn downscale_mode(image: &ArrayView3<u8>, scale: usize) -> Array3<u8> {
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
                                    // Pack RGB into 24-bit integer for fast hashing
                                    let r = image[(y, x, 0)] as u32;
                                    let g = image[(y, x, 1)] as u32;
                                    let b = image[(y, x, 2)] as u32;
                                    let color_key = (r << 16) | (g << 8) | b;
                                    *color_counts.entry(color_key).or_insert(0) += 1;
                                }
                            } else {
                                let r = image[(y, x, 0)] as u32;
                                let g = image[(y, x, 1)] as u32;
                                let b = image[(y, x, 2)] as u32;
                                let color_key = (r << 16) | (g << 8) | b;
                                *color_counts.entry(color_key).or_insert(0) += 1;
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

                    if !color_counts.is_empty() {
                        // Find most frequent color (mode)
                        let most_frequent_color = color_counts
                            .iter()
                            .max_by_key(|(_, &count)| count)
                            .map(|(&color, _)| color)
                            .unwrap_or(0);

                        pixel[0] = ((most_frequent_color >> 16) & 0xFF) as u8;
                        pixel[1] = ((most_frequent_color >> 8) & 0xFF) as u8;
                        pixel[2] = (most_frequent_color & 0xFF) as u8;

                        if !has_alpha {
                            pixel[3] = 255;
                        }
                    } else {
                        // No pixels - set to black
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

// Enhanced K-Means with k-means++ initialization to match scikit-learn
struct KMeansResult {
    centroids: Vec<[f32; 3]>,
    labels: Vec<usize>,
}

fn kmeans_plusplus(points: &Vec<[f32; 3]>, k: usize, rng: &mut StdRng) -> Vec<[f32; 3]> {
    let n = points.len();
    if k >= n {
        // Return all points if k >= n
        return points.iter().take(k).cloned().collect();
    }

    let mut centroids = Vec::with_capacity(k);
    
    // Choose first centroid randomly
    let first_idx = rng.gen_range(0..n);
    centroids.push(points[first_idx]);
    
    // Choose remaining centroids
    for _ in 1..k {
        // Calculate distances to nearest centroid for each point
        let mut distances = Vec::with_capacity(n);
        for point in points {
            let mut min_dist = f32::MAX;
            for centroid in &centroids {
                let dist = euclidean_distance_squared(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            distances.push(min_dist);
        }
        
        // Choose next centroid with probability proportional to distance^2
        let total_weight: f32 = distances.iter().sum();
        if total_weight <= 0.0 {
            // All points are at same location, pick randomly
            let idx = rng.gen_range(0..n);
            centroids.push(points[idx]);
        } else {
            let mut rand_val: f32 = rng.gen();
            rand_val *= total_weight;
            
            let mut cumulative = 0.0;
            let mut chosen_idx = 0;
            for (i, &weight) in distances.iter().enumerate() {
                cumulative += weight;
                if cumulative >= rand_val {
                    chosen_idx = i;
                    break;
                }
            }
            centroids.push(points[chosen_idx]);
        }
    }
    
    centroids
}

fn euclidean_distance_squared(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

fn euclidean_distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    euclidean_distance_squared(a, b).sqrt()
}

fn kmeans_with_seed(points: &Vec<[f32; 3]>, k: usize, max_iters: usize, seed: u64) -> Option<KMeansResult> {
    if points.is_empty() || k == 0 {
        return None;
    }

    if k >= points.len() {
        // Special case: each point is its own cluster
        let mut labels = Vec::with_capacity(points.len());
        let mut centroids = Vec::with_capacity(k);
        
        for i in 0..points.len() {
            labels.push(i);
            centroids.push(points[i]);
        }
        // Fill remaining centroids if needed
        while centroids.len() < k {
            centroids.push(points[0]); // pad with first point
            labels.push(0);
        }
        
        return Some(KMeansResult { centroids, labels });
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = kmeans_plusplus(points, k, &mut rng);
    
    let mut labels = vec![0; points.len()];
    let mut prev_labels = vec![usize::MAX; points.len()];
    
    for _iter in 0..max_iters {
        // Assignment step
        let mut changed = false;
        for (i, point) in points.iter().enumerate() {
            let mut min_dist = f32::MAX;
            let mut best_centroid = 0;
            
            for (j, centroid) in centroids.iter().enumerate() {
                let dist = euclidean_distance_squared(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_centroid = j;
                }
            }
            
            if labels[i] != best_centroid {
                changed = true;
                labels[i] = best_centroid;
            }
        }
        
        if !changed {
            break;
        }
        
        // Update step
        let mut new_centroids = vec![[0.0, 0.0, 0.0]; k];
        let mut counts = vec![0; k];
        
        for (i, &label) in labels.iter().enumerate() {
            if label < k {
                new_centroids[label][0] += points[i][0];
                new_centroids[label][1] += points[i][1];
                new_centroids[label][2] += points[i][2];
                counts[label] += 1;
            }
        }
        
        // Calculate new centroids
        for i in 0..k {
            if counts[i] > 0 {
                new_centroids[i][0] /= counts[i] as f32;
                new_centroids[i][1] /= counts[i] as f32;
                new_centroids[i][2] /= counts[i] as f32;
            } else {
                // Keep old centroid if no points assigned
                new_centroids[i] = centroids[i];
            }
        }
        
        centroids = new_centroids;
    }
    
    Some(KMeansResult { centroids, labels })
}

/// Extremely fast dominant color downscaling using KMeans
pub fn downscale_dominant(image: &ArrayView3<u8>, scale: usize, threshold: f32) -> Array3<u8> {
    let (height, width, channels) = image.dim();
    let target_h = height / scale;
    let target_w = width / scale;
    let has_alpha = channels == 4;

    // Initialize result array
    let mut result = Array3::<u8>::zeros((target_h, target_w, channels));

    // --- Pre-process Alpha Channel ---
    if has_alpha {
        for ty in 0..target_h {
            for tx in 0..target_w {
                let y_start = ty * scale;
                let x_start = tx * scale;
                let y_end = (ty + 1) * scale;
                let x_end = (tx + 1) * scale;

                // Collect alpha values for this block
                let mut alpha_values = Vec::with_capacity(scale * scale);
                for y in y_start..y_end {
                    for x in x_start..x_end {
                        alpha_values.push(image[(y, x, 3)]);
                    }
                }

                // Calculate median
                alpha_values.sort_unstable();
                let median_alpha = alpha_values[alpha_values.len() / 2];
                result[(ty, tx, 3)] = if median_alpha > 128 { 255 } else { 0 };
            }
        }
    }

    // --- Process RGB channels ---
    let results: Vec<_> = (0..target_h)
        .into_par_iter()
        .flat_map(|ty| {
            (0..target_w)
                .into_par_iter()
                .map(move |tx| {
                    let y_start = ty * scale;
                    let x_start = tx * scale;
                    let y_end = (ty + 1) * scale;
                    let x_end = (tx + 1) * scale;

                    // Extract block pixels
                    let mut color_pixels = Vec::new();
                    let mut has_opaque_pixels = false;

                    if has_alpha {
                        // Collect only opaque pixels
                        for y in y_start..y_end {
                            for x in x_start..x_end {
                                if image[(y, x, 3)] > 128 {
                                    color_pixels.push([
                                        image[(y, x, 0)] as f32,
                                        image[(y, x, 1)] as f32,
                                        image[(y, x, 2)] as f32,
                                    ]);
                                    has_opaque_pixels = true;
                                }
                            }
                        }
                    } else {
                        // Collect all pixels
                        for y in y_start..y_end {
                            for x in x_start..x_end {
                                color_pixels.push([
                                    image[(y, x, 0)] as f32,
                                    image[(y, x, 1)] as f32,
                                    image[(y, x, 2)] as f32,
                                ]);
                            }
                        }
                        has_opaque_pixels = !color_pixels.is_empty();
                    }

                    // If no opaque pixels and has alpha, result is already [0,0,0,0] from alpha pre-processing
                    if !has_opaque_pixels {
                        return (ty, tx, None);
                    }

                    // If only one pixel, use it directly
                    if color_pixels.len() == 1 {
                        let pixel = [
                            color_pixels[0][0] as u8,
                            color_pixels[0][1] as u8,
                            color_pixels[0][2] as u8,
                        ];
                        return (ty, tx, Some(pixel));
                    }

                    // Find unique colors for cluster count determination
                    let mut unique_colors = HashMap::new();
                    for pixel in &color_pixels {
                        let key = (pixel[0] as u32, pixel[1] as u32, pixel[2] as u32);
                        unique_colors.insert(key, ());
                    }
                    let num_unique = unique_colors.len();

                    // Handle single unique color
                    if num_unique == 1 {
                        let pixel = [
                            color_pixels[0][0] as u8,
                            color_pixels[0][1] as u8,
                            color_pixels[0][2] as u8,
                        ];
                        return (ty, tx, Some(pixel));
                    }

                    // Determine number of clusters
                    let n_clusters = std::cmp::min(3, std::cmp::max(1, num_unique / 2));

                    // Perform K-Means clustering with same parameters as Python
                    // random_state=0, n_init=1
                    match kmeans_with_seed(&color_pixels, n_clusters, 50, 0) {
                        Some(kmeans_result) => {
                            // Count cluster sizes
                            let mut cluster_sizes = vec![0; n_clusters];
                            for &label in &kmeans_result.labels {
                                if label < n_clusters {
                                    cluster_sizes[label] += 1;
                                }
                            }

                            // Find dominant cluster
                            let dominant_cluster_idx = cluster_sizes
                                .iter()
                                .enumerate()
                                .max_by_key(|(_, &size)| size)
                                .map(|(idx, _)| idx)
                                .unwrap_or(0);
                            
                            let dominant_cluster_size = cluster_sizes[dominant_cluster_idx];
                            let total_pixels = color_pixels.len();

                            // Check dominance threshold
                            if (dominant_cluster_size as f32) / (total_pixels as f32) >= threshold {
                                // Use dominant cluster center
                                let center = &kmeans_result.centroids[dominant_cluster_idx];
                                let pixel = [
                                    center[0].round().clamp(0.0, 255.0) as u8,
                                    center[1].round().clamp(0.0, 255.0) as u8,
                                    center[2].round().clamp(0.0, 255.0) as u8,
                                ];
                                (ty, tx, Some(pixel))
                            } else {
                                // Fallback to mean color
                                let mut r_sum = 0.0;
                                let mut g_sum = 0.0;
                                let mut b_sum = 0.0;
                                for pixel in &color_pixels {
                                    r_sum += pixel[0];
                                    g_sum += pixel[1];
                                    b_sum += pixel[2];
                                }
                                let count = color_pixels.len() as f32;
                                let pixel = [
                                    (r_sum / count).round().clamp(0.0, 255.0) as u8,
                                    (g_sum / count).round().clamp(0.0, 255.0) as u8,
                                    (b_sum / count).round().clamp(0.0, 255.0) as u8,
                                ];
                                (ty, tx, Some(pixel))
                            }
                        }
                        None => {
                            // Fallback to mean color if K-Means fails
                            let mut r_sum = 0.0;
                            let mut g_sum = 0.0;
                            let mut b_sum = 0.0;
                            for pixel in &color_pixels {
                                r_sum += pixel[0];
                                g_sum += pixel[1];
                                b_sum += pixel[2];
                            }
                            let count = color_pixels.len() as f32;
                            let pixel = [
                                (r_sum / count).round().clamp(0.0, 255.0) as u8,
                                (g_sum / count).round().clamp(0.0, 255.0) as u8,
                                (b_sum / count).round().clamp(0.0, 255.0) as u8,
                            ];
                            (ty, tx, Some(pixel))
                        }
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Apply RGB results
    for (ty, tx, rgb_opt) in results {
        if let Some(rgb) = rgb_opt {
            result[(ty, tx, 0)] = rgb[0];
            result[(ty, tx, 1)] = rgb[1];
            result[(ty, tx, 2)] = rgb[2];
            
            // Set alpha to 255 if no alpha channel
            if !has_alpha {
                result[(ty, tx, 3)] = 255;
            }
        } else if !has_alpha {
            // No opaque pixels and no alpha channel - set to black with full opacity
            result[(ty, tx, 0)] = 0;
            result[(ty, tx, 1)] = 0;
            result[(ty, tx, 2)] = 0;
            result[(ty, tx, 3)] = 255;
        }
    }

    result
}