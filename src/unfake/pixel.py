# File: pixel.py
#!/usr/bin/env python3
"""
Forked from unfake.py by Benjamin Paine (github @painebenjamin)
Based on unfake.js by Eugeniy Smirnov (github @jenissimo)
"""


import json
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import warnings
from numpy.typing import NDArray
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning  # Proper import for the warning

# Suppress the specific KMeans convergence warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from .content_adaptive import content_adaptive_downscale
from .wu_quantizer import WuQuantizer

# Import accelerated versions when available
try:
    from .pixel_rust_integration import (
        RUST_AVAILABLE,
        WuQuantizerAccelerated,
        count_colors_accelerated,
        downscale_dominant_color_accelerated,
        downscale_mode_accelerated,
        finalize_pixels_accelerated,
        map_pixels_to_palette_accelerated,
        runs_based_detect_accelerated,
    )
except ImportError:
    RUST_AVAILABLE = False
    logger = logging.getLogger("pixel.py")
    logger.info("Rust acceleration not available, using Python implementations")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unfake.py")


@dataclass
class ProcessingManifest:
    """Stores metadata about the image processing pipeline"""

    original_size: Tuple[int, int]
    final_size: Tuple[int, int]
    processing_steps: Dict
    processing_time_ms: int
    timestamp: str


def gcd_array(arr: List[int]) -> int:
    """Calculate GCD of an array of numbers"""
    if not arr:
        return 1
    result = arr[0]
    for i in range(1, len(arr)):
        result = math.gcd(result, arr[i])
        if result == 1:
            return 1
    return result


def detect_scale_from_signal(signal: np.ndarray) -> int:
    """Detect pixel grid scale from a signal using peak analysis"""
    if len(signal) < 3:
        return 1

    mean_val = np.mean(signal)
    std_val = np.std(signal)
    threshold = mean_val + 1.5 * std_val

    # Find peaks
    peaks: List[int] = []
    for i in range(1, len(signal) - 1):
        if signal[i] > threshold and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if not peaks or i - peaks[-1] > 2:
                peaks.append(i)

    if len(peaks) <= 2:
        logger.debug("Not enough peaks found, returning 1")
        return 1

    # Calculate spacings between peaks
    spacings = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
    logger.debug(f"Found {len(peaks)} peaks, spacings: {spacings}")

    # Use median spacing if consistent
    median_spacing = int(np.median(spacings))
    close_spacings = [s for s in spacings if abs(s - median_spacing) <= 2]

    if len(close_spacings) / len(spacings) > 0.7:
        logger.debug(f"Using median spacing: {median_spacing}")
        return max(1, median_spacing)

    # Otherwise use mode
    mode_spacing = Counter(spacings).most_common(1)[0][0]
    logger.debug(f"Using mode spacing: {mode_spacing}")
    return max(1, mode_spacing)


def runs_based_detect(image: np.ndarray) -> int:
    """Detect pixel art scale by analyzing color run lengths"""
    h, w = image.shape[:2]
    all_run_lengths = []

    # Scan horizontal runs
    for y in range(h):
        run_length = 1
        for x in range(1, w):
            if np.array_equal(image[y, x], image[y, x - 1]):
                run_length += 1
            else:
                if run_length > 1:
                    all_run_lengths.append(run_length)
                run_length = 1
        if run_length > 1:
            all_run_lengths.append(run_length)

    # Scan vertical runs
    for x in range(w):
        run_length = 1
        for y in range(1, h):
            if np.array_equal(image[y, x], image[y - 1, x]):
                run_length += 1
            else:
                if run_length > 1:
                    all_run_lengths.append(run_length)
                run_length = 1
        if run_length > 1:
            all_run_lengths.append(run_length)

    if len(all_run_lengths) < 10:
        return 1

    detected_scale = gcd_array(all_run_lengths)
    logger.info(f"Runs-based detection found scale: {detected_scale}")
    return max(1, detected_scale)


def edge_aware_detect(image: np.ndarray) -> int:
    """Detect pixel art scale using edge-aware algorithm"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY if image.shape[2] == 4 else cv2.COLOR_RGB2GRAY)

    # Use tiled approach for large images
    h, w = gray.shape
    all_scales = []

    TILE_COUNT = 3
    OVERLAP = 0.25

    tile_w = w // TILE_COUNT
    tile_h = h // TILE_COUNT
    overlap_w = int(tile_w * OVERLAP)
    overlap_h = int(tile_h * OVERLAP)

    if tile_w < 50 or tile_h < 50:
        # Image too small for tiling, use single ROI
        return single_region_edge_detect(gray)

    for y in range(TILE_COUNT):
        for x in range(TILE_COUNT):
            roi_x = max(0, x * tile_w - overlap_w)
            roi_y = max(0, y * tile_h - overlap_h)
            roi_w = min(w - roi_x, tile_w + 2 * overlap_w)
            roi_h = min(h - roi_y, tile_h + 2 * overlap_h)

            if roi_w < 30 or roi_h < 30:
                continue

            tile = gray[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

            # Check variance
            mean, stddev = cv2.meanStdDev(tile)
            if stddev[0][0] < 5.0:
                logger.debug(f"Skipping tile ({x},{y}) due to low variance")
                continue

            # Get gradient profiles
            h_profile = get_gradient_profile(tile, "horizontal")
            v_profile = get_gradient_profile(tile, "vertical")

            h_scale = detect_scale_from_signal(h_profile)
            v_scale = detect_scale_from_signal(v_profile)

            if h_scale > 1:
                all_scales.append(h_scale)
            if v_scale > 1:
                all_scales.append(v_scale)

    if not all_scales:
        logger.warning("Tiled detection found no scales, trying single region")
        return single_region_edge_detect(gray)

    # Return mode of detected scales
    scale_counts = Counter(all_scales)
    best_scale = scale_counts.most_common(1)[0][0]
    logger.info(f"Edge-aware detection found scales: {all_scales}, best: {best_scale}")
    return best_scale


def single_region_edge_detect(gray: np.ndarray) -> int:
    """Detect scale on a single region"""
    h, w = gray.shape

    # Use center 75% as ROI
    roi_x = int(w * 0.125)
    roi_y = int(h * 0.125)
    roi_w = int(w * 0.75)
    roi_h = int(h * 0.75)

    if roi_w < 3 or roi_h < 3:
        return 1

    roi = gray[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

    h_profile = get_gradient_profile(roi, "horizontal")
    v_profile = get_gradient_profile(roi, "vertical")

    h_scale = detect_scale_from_signal(h_profile)
    v_scale = detect_scale_from_signal(v_profile)

    if h_scale > 1 and v_scale > 1 and abs(h_scale - v_scale) <= 2:
        return round((h_scale + v_scale) / 2)

    return max(h_scale, v_scale, 1)


def get_gradient_profile(gray: np.ndarray, direction: str) -> np.ndarray:
    """Generate gradient profile for scale detection"""
    if direction == "horizontal":
        sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        profile = np.sum(np.abs(sobel), axis=0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        profile = np.sum(np.abs(sobel), axis=1)
    return profile  # type: ignore[no-any-return]


def find_optimal_crop(gray: np.ndarray, scale: int) -> Tuple[int, int]:
    """Find optimal crop offset to align with pixel grid"""
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    profile_x = np.sum(np.abs(sobel_x), axis=0)
    profile_y = np.sum(np.abs(sobel_y), axis=1)

    def find_best_offset(profile: np.ndarray, s: int) -> int:
        best_offset = 0
        max_score: float = -1.0
        for offset in range(s):
            score: float = np.sum(profile[offset::s])
            if score > max_score:
                max_score = score
                best_offset = offset
        return best_offset

    best_dx = find_best_offset(profile_x, scale)
    best_dy = find_best_offset(profile_y, scale)

    logger.info(f"Optimal crop found: x={best_dx}, y={best_dy}")
    return best_dx, best_dy


def alpha_binarization(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Convert alpha channel to binary (0 or 255)"""
    if image.shape[2] < 4:
        return image

    result = image.copy()
    result[:, :, 3] = np.where(result[:, :, 3] >= threshold, 255, 0)
    return result  # type: ignore[no-any-return]


def morphological_cleanup(image: np.ndarray) -> np.ndarray:
    """Apply morphological operations to clean up the image"""
    kernel = np.ones((2, 2), np.uint8)

    # Extract alpha channel for processing
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        # OPEN to remove noise
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
        # CLOSE to fill gaps
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

        result = image.copy()
        result[:, :, 3] = alpha
        return result  # type: ignore[no-any-return]
    else:
        # Process each channel
        result = image.copy()
        for i in range(3):
            result[:, :, i] = cv2.morphologyEx(result[:, :, i], cv2.MORPH_OPEN, kernel)
            result[:, :, i] = cv2.morphologyEx(result[:, :, i], cv2.MORPH_CLOSE, kernel)
        return result  # type: ignore[no-any-return]


def jaggy_cleaner(image: np.ndarray) -> np.ndarray:
    """Remove isolated diagonal pixels"""
    h, w = image.shape[:2]
    result = image.copy()

    def is_opaque(y: int, x: int) -> bool:
        if 0 <= y < h and 0 <= x < w:
            return result[y, x, 3] > 128 if image.shape[2] == 4 else True
        return False

    # Check each pixel
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not is_opaque(y, x):
                continue

            # Count orthogonal and diagonal neighbors
            n = is_opaque(y - 1, x)
            s = is_opaque(y + 1, x)
            e = is_opaque(y, x + 1)
            w = is_opaque(y, x - 1)
            ne = is_opaque(y - 1, x + 1)
            nw = is_opaque(y - 1, x - 1)
            se = is_opaque(y + 1, x + 1)
            sw = is_opaque(y + 1, x - 1)

            orth_count: int = sum([n, s, e, w])
            diag_count: int = sum([ne, nw, se, sw])

            # Remove if no orthogonal neighbors and only one diagonal
            if orth_count == 0 and diag_count == 1:
                if image.shape[2] == 4:
                    result[y, x] = [0, 0, 0, 0]
                else:
                    result[y, x] = [0, 0, 0]

    return result  # type: ignore[no-any-return]


def finalize_pixels(image: np.ndarray) -> np.ndarray:
    """Ensure pixels have binary alpha and transparent pixels are black"""
    # Use accelerated version if available
    if "RUST_AVAILABLE" in globals() and RUST_AVAILABLE:
        return finalize_pixels_accelerated(image)

    # Fallback to Python implementation
    if image.shape[2] < 4:
        return image

    result = image.copy()
    mask = result[:, :, 3] < 128
    result[mask] = [0, 0, 0, 0]
    result[~mask, 3] = 255
    return result  # type: ignore[no-any-return]


def pre_downscale_filter(image: np.ndarray) -> np.ndarray:
    """Apply noise/AA reduction filter"""
    # Gaussian blur
    blurred = cv2.GaussianBlur(image, (3, 3), 0.5)
    # Unsharp masking
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened


def post_downscale_sharpen(image: np.ndarray) -> np.ndarray:
    """Apply selective sharpening"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY if image.shape[2] == 4 else cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    var_map = cv2.convertScaleAbs(laplacian)
    high_var_mask = var_map > 50  # Threshold for high variance areas

    # Apply median filter on high variance areas
    result = image.copy()
    for c in range(image.shape[2]):
        channel = result[:, :, c]
        median = cv2.medianBlur(channel, 3)
        channel[high_var_mask] = median[high_var_mask]
    return result

def downscale_by_dominant_color(
    image: np.ndarray, scale: int, threshold: float = 0.05
) -> np.ndarray:
    """
    Optimized downscale using dominant color method with KMeans clustering (best for pixel art).
    Prioritizes speed while maintaining logic and quality.
    """
    h, w = image.shape[:2]
    target_h = h // scale
    target_w = w // scale
    has_alpha = image.shape[2] == 4
    channels = 4 if has_alpha else 3

    # Initialize result array
    result = np.zeros((target_h, target_w, channels), dtype=np.uint8)

    # --- Pre-process Alpha Channel ---
    if has_alpha:
        # Reshape image alpha channel for easier block processing
        alpha_reshaped = image[:, :, 3].reshape(target_h, scale, target_w, scale)
        # Calculate median alpha per block efficiently
        alpha_medians = np.median(alpha_reshaped, axis=(1, 3))
        # Binarize alpha for the entire result at once
        result[:, :, 3] = np.where(alpha_medians > 128, 255, 0)

    # --- Vectorized Block Extraction ---
    # Reshape the image into blocks of (scale x scale)
    # New shape: (target_h, scale, target_w, scale, channels)
    blocks = image[:target_h*scale, :target_w*scale].reshape(target_h, scale, target_w, scale, channels)

    # Process RGB channels for all blocks at once where possible
    for ty in range(target_h):
        for tx in range(target_w):
            block = blocks[ty, :, tx, :, :]

            if has_alpha:
                # --- Handle Alpha Channel Logic ---
                # Consider only opaque pixels for color determination
                opaque_mask = block[:, :, 3] > 128
                if not np.any(opaque_mask):
                    # No opaque pixels, result is already [0,0,0,0] due to alpha pre-processing
                    continue
                color_pixels = block[opaque_mask][:, :3] # Get RGB of opaque pixels
            else:
                # No alpha, consider all pixels in the block
                color_pixels = block.reshape(-1, 3)

            # --- Color Processing ---
            # Handle case with no opaque pixels (already handled above for alpha)
            if color_pixels.size == 0:
                if not has_alpha:
                    result[ty, tx] = [0, 0, 0]
                continue # Move to next block

            # --- Find Unique Colors ---
            # Use unique on the relevant pixel subset (opaque or all)
            unique_colors, counts = np.unique(color_pixels, axis=0, return_counts=True)
            num_unique = len(unique_colors)

            # Handle single unique color
            if num_unique == 1:
                result[ty, tx, :3] = unique_colors[0]
                continue # Move to next block

            # --- KMeans Clustering for Multiple Colors ---
            # Determine number of clusters based on unique colors
            n_clusters = min(3, max(1, num_unique // 2))

            # Perform KMeans clustering
            # We keep KMeans in the loop as it's hard to vectorize across differently sized blocks.
            # However, we optimize by fitting only on the necessary data and using minimal n_init.
            kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=0)
            kmeans.fit(color_pixels.astype(float))

            # Find the dominant cluster
            cluster_sizes = np.bincount(kmeans.labels_, minlength=n_clusters)
            dominant_cluster_idx = np.argmax(cluster_sizes)
            dominant_cluster_size = cluster_sizes[dominant_cluster_idx]

            # Check dominance threshold
            total_pixels = len(color_pixels)
            if dominant_cluster_size / total_pixels >= threshold:
                # Use dominant cluster center
                dominant_color = kmeans.cluster_centers_[dominant_cluster_idx]
                result[ty, tx, :3] = np.clip(dominant_color, 0, 255).astype(np.uint8)
            else:
                # Fallback to mean color
                mean_color = np.mean(color_pixels, axis=0)
                result[ty, tx, :3] = np.clip(mean_color, 0, 255).astype(np.uint8)

    return result

def downscale_block(image: np.ndarray, scale: int, method: str = "median") -> np.ndarray:
    """Downscale using various aggregation methods"""
    # Use accelerated mode if available and method is "mode"
    if method == "mode" and "RUST_AVAILABLE" in globals() and RUST_AVAILABLE:
        return downscale_mode_accelerated(image, scale)

    h, w = image.shape[:2]
    target_h = h // scale
    target_w = w // scale

    has_alpha = image.shape[2] == 4
    channels = 4 if has_alpha else 3
    result = np.zeros((target_h, target_w, channels), dtype=np.uint8)

    for ty in range(target_h):
        for tx in range(target_w):
            block = image[ty * scale : (ty + 1) * scale, tx * scale : (tx + 1) * scale]

            if has_alpha:
                # Handle alpha separately
                alpha_values = block[:, :, 3].flatten()
                result[ty, tx, 3] = 255 if np.median(alpha_values) > 128 else 0

                # Only process opaque pixels for color
                opaque_mask = block[:, :, 3] > 128
                if np.any(opaque_mask):
                    opaque_pixels = block[opaque_mask][:, :3]

                    if method == "median":
                        result[ty, tx, :3] = np.median(opaque_pixels, axis=0)
                    elif method == "mean":
                        result[ty, tx, :3] = np.mean(opaque_pixels, axis=0)
                    elif method == "mode":
                        # Find most common color
                        pixels_tuple = [tuple(p) for p in opaque_pixels]
                        most_common = Counter(pixels_tuple).most_common(1)[0][0]
                        result[ty, tx, :3] = most_common
                    elif method == "nearest":
                        result[ty, tx, :3] = opaque_pixels[0]
                else:
                    result[ty, tx] = [0, 0, 0, 0]
            else:
                pixels = block.reshape(-1, 3)
                if method == "median":
                    result[ty, tx] = np.median(pixels, axis=0)
                elif method == "mean":
                    result[ty, tx] = np.mean(pixels, axis=0)
                elif method == "mode":
                    pixels_tuple = [tuple(p) for p in pixels]
                    most_common = Counter(pixels_tuple).most_common(1)[0][0]
                    result[ty, tx] = most_common
                elif method == "nearest":
                    result[ty, tx] = pixels[0]

    return result  # type: ignore[no-any-return]


def hybrid_downscale(image: np.ndarray, scale: int, threshold: float = 0.05) -> np.ndarray:
    """Hybrid downscale: content-adaptive for high variance, dominant for low"""
    h, w = image.shape[:2]
    target_h = h // scale
    target_w = w // scale

    # Compute full content-adaptive
    adaptive = content_adaptive_downscale(image, target_w, target_h)

    # Compute variance map on original
    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY if image.shape[2] == 4 else cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    var_map = np.abs(laplacian)
    var_map_resized = cv2.resize(var_map, (target_w, target_h), interpolation=cv2.INTER_AREA)
    high_var_mask = var_map_resized > np.mean(var_map_resized)  # Above average variance

    # Compute dominant
    if "RUST_AVAILABLE" in globals() and RUST_AVAILABLE:
        dominant = downscale_dominant_color_accelerated(image, scale, threshold)
    else:
        dominant = downscale_by_dominant_color(image, scale, threshold) 

    # Blend: adaptive on high var, dominant on low
    result = dominant.copy()
    result[high_var_mask] = adaptive[high_var_mask]

    return result


def edge_preserving_refinement(image: np.ndarray) -> np.ndarray:
    """Apply edge-preserving refinement"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY if image.shape[2] == 4 else cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8))

    # Majority filter on edges
    majority = cv2.medianBlur(image, 3)  # Simple majority approx

    result = image.copy()
    result[edges_dilated > 0] = majority[edges_dilated > 0]

    return result


def quantize_colors(
    image: np.ndarray, max_colors: int, fixed_palette: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Quantize image colors using Wu algorithm or fixed palette"""
    if fixed_palette:
        # Convert hex colors to RGB
        palette_rgb = []
        for hex_color in fixed_palette:
            hex_color = hex_color.lstrip("#")
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            palette_rgb.append((r, g, b))

        # Use accelerated palette mapping if available
        if "RUST_AVAILABLE" in globals() and RUST_AVAILABLE:
            result = map_pixels_to_palette_accelerated(image, palette_rgb)
        else:
            # Fallback to Python implementation
            h, w = image.shape[:2]
            result = np.zeros_like(image)

            for y in range(h):
                for x in range(w):
                    if image.shape[2] == 4 and image[y, x, 3] < 128:
                        result[y, x] = [0, 0, 0, 0]
                    else:
                        pixel = image[y, x, :3]
                        min_dist = float("inf")
                        best_color = palette_rgb[0]

                        for color in palette_rgb:
                            dist: float = np.sum((pixel - color) ** 2)
                            if dist < min_dist:
                                min_dist = dist
                                best_color = color

                        result[y, x, :3] = best_color
                        if image.shape[2] == 4:
                            result[y, x, 3] = 255

        return result, palette_rgb
    else:
        # Use accelerated Wu quantization if available
        if "RUST_AVAILABLE" in globals() and RUST_AVAILABLE:
            quantizer = WuQuantizerAccelerated(max_colors)
        else:
            quantizer = WuQuantizer(max_colors)  # type: ignore[assignment]
        return quantizer.quantize(image)


def extract_palette(image: np.ndarray) -> List[str]:
    """Extract color palette from image as hex strings"""
    h, w = image.shape[:2]
    unique_colors = set()

    for y in range(h):
        for x in range(w):
            if image.shape[2] == 4 and image[y, x, 3] < 128:
                continue
            color = tuple(image[y, x, :3])
            unique_colors.add(color)

    # Convert to hex
    palette = []
    for r, g, b in unique_colors:
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        palette.append(hex_color)

    return sorted(palette)


def count_colors(image: np.ndarray) -> int:
    """Count unique opaque colors in image"""
    # Use accelerated version if available
    if "RUST_AVAILABLE" in globals() and RUST_AVAILABLE:
        return count_colors_accelerated(image)

    # Fallback to Python implementation
    if image.shape[2] == 4:
        opaque_mask = image[:, :, 3] >= 128
        opaque_pixels = image[opaque_mask][:, :3]
    else:
        opaque_pixels = image.reshape(-1, 3)

    if len(opaque_pixels) == 0:
        return 0

    unique_colors = np.unique(opaque_pixels, axis=0)
    return len(unique_colors)


def detect_optimal_color_count(
    image: np.ndarray,
    downsample_to: int = 64,
    color_quantize_factor: int = 48,
    dominance_threshold: float = 0.015,
    max_colors: int = 32,
) -> int:
    """
    Automatically detect optimal number of colors in an image
    Uses aggressive clustering and dominant color analysis

    Args:
        image: Input image
        downsample_to: Size to downsample for analysis
        color_quantize_factor: Factor for color quantization
        dominance_threshold: Threshold for dominant colors
        max_colors: Maximum allowed colors

    Returns:
        Optimal number of colors
    """
    logger.info("Detecting optimal color count...")

    # Downsample for faster analysis
    h, w = image.shape[:2]
    aspect_ratio = h / w
    target_width = downsample_to
    target_height = int(target_width * aspect_ratio)

    small_img = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Apply blur to remove noise and gradients
    small_img = cv2.medianBlur(small_img, 5)
    small_img = cv2.GaussianBlur(small_img, (5, 5), 1)

    # Gather color statistics with aggressive quantization
    color_counts: Dict[str, int] = {}
    total_pixels = 0

    for y in range(small_img.shape[0]):
        for x in range(small_img.shape[1]):
            if small_img.shape[2] == 4 and small_img[y, x, 3] < 200:
                continue  # Skip transparent pixels

            r = int(small_img[y, x, 0] / color_quantize_factor) * color_quantize_factor
            g = int(small_img[y, x, 1] / color_quantize_factor) * color_quantize_factor
            b = int(small_img[y, x, 2] / color_quantize_factor) * color_quantize_factor

            color_key = f"{r},{g},{b}"
            color_counts[color_key] = color_counts.get(color_key, 0) + 1
            total_pixels += 1

    # Analyze for dominant colors
    min_pixels_for_dominance = max(3, int(total_pixels * dominance_threshold))
    logger.info(f"Found {len(color_counts)} unique quantized colors (pre-filter)")

    significant_colors = [
        color for color, count in color_counts.items() if count >= min_pixels_for_dominance
    ]

    # If too many colors, apply stricter threshold
    if len(significant_colors) > max_colors:
        strict_threshold = max(min_pixels_for_dominance, int(total_pixels * 0.02))
        significant_colors = [
            color for color, count in color_counts.items() if count >= strict_threshold
        ]
        logger.info(f"Filtered down to {len(significant_colors)} dominant colors")

    result = max(2, min(len(significant_colors), max_colors))
    logger.info(f"Final auto-detected color count: {result}")
    return result  # type: ignore[no-any-return]


async def process_image(
    file_path_or_image: Union[str, Image.Image, NDArray],
    max_colors: Optional[int] = None,  # None for auto-detection
    manual_scale: Optional[Union[int, List[int]]] = None,
    detect_method: str = "auto",  # 'auto', 'runs', 'edge'
    downscale_method: str = "dominant",  # 'dominant', 'median', 'mode', 'mean', 'nearest', 'content-adaptive', 'hybrid'
    dom_mean_threshold: float = 0.05,
    cleanup: Optional[Dict[str, bool]] = None,
    fixed_palette: Optional[List[str]] = None,
    alpha_threshold: int = 128,
    snap_grid: bool = True,
    auto_color_detect: bool = False,
    pre_filter: bool = False,
    edge_preserve: bool = False,
    post_sharpen: bool = False,
    iterations: int = 1,
    file_path: Optional[str] = None,  # deprecated
) -> Dict:
    """
    Main image processing pipeline

    Args:
        file_path_or_image: Path to input image, PIL image, or numpy array
        max_colors: Maximum number of colors in output (None for auto-detection)
        manual_scale: Manual scale override
        detect_method: Scale detection method ('auto', 'runs', 'edge')
        downscale_method: Downscaling algorithm ('dominant', 'median', 'mode', 'mean', 'nearest', 'content-adaptive', 'hybrid')
        dom_mean_threshold: Threshold for dominant color method
        cleanup: Cleanup options dict with 'morph' and 'jaggy' keys
        fixed_palette: Optional fixed color palette (hex strings)
        alpha_threshold: Alpha binarization threshold (0-255)
        snap_grid: Whether to snap to pixel grid
        auto_color_detect: Force automatic color detection
        pre_filter: Apply pre-downscale filter
        edge_preserve: Apply edge-preserving refinement for content-adaptive
        post_sharpen: Apply post-downscale sharpening
        iterations: Number of iterations for rerun
        file_path: Path to input image (deprecated)

    Returns:
        Dictionary with processed image data, palette, and manifest
    """
    if cleanup is None:
        cleanup = {"morph": False, "jaggy": False}

    start_time = time.time()

    if file_path is not None:
        logger.warning("file_path is deprecated, use file_path_or_image instead")
        file_path_or_image = file_path

    if isinstance(file_path_or_image, str):
        # Load image
        pil_image = Image.open(file_path_or_image)
        if pil_image.mode != "RGBA":
            pil_image = pil_image.convert("RGBA")  # type: ignore[assignment]
        current = np.array(pil_image)
    elif isinstance(file_path_or_image, Image.Image):
        if file_path_or_image.mode != "RGBA":
            file_path_or_image = file_path_or_image.convert("RGBA")
        current = np.array(file_path_or_image)
    elif isinstance(file_path_or_image, np.ndarray):
        if file_path_or_image.ndim == 4:
            assert file_path_or_image.shape[0] == 1, "Batch dimension is not supported"
            file_path_or_image = file_path_or_image[0]

        if file_path_or_image.dtype != np.uint8:
            file_path_or_image = (file_path_or_image * 255).astype(np.uint8)

        h, w, c = file_path_or_image.shape
        if c == 4:
            current = file_path_or_image
        elif c == 3:
            current = np.concatenate([
                file_path_or_image,
                (np.ones((h, w, 1)) * 255).astype(np.uint8),
            ], axis=2)
        else:
            raise ValueError(f"Unsupported number of channels: {c}")

    original_size = (current.shape[1], current.shape[0])

    logger.info(f"Processing image: {original_size[0]}x{original_size[1]}")

    # Check size limits
    if (
        current.shape[0] > 8000
        or current.shape[1] > 8000
        or (current.shape[0] * current.shape[1] > 10_000_000)
    ):
        raise ValueError(f"Image too large: {current.shape[1]}x{current.shape[0]}")

    # 1. Pre-processing: Binarize alpha
    if alpha_threshold is not None:
        current = alpha_binarization(current, alpha_threshold)

    # 2. Scale Detection
    scale = 1
    if manual_scale:
        scale = max(1, manual_scale[0] if isinstance(manual_scale, list) else manual_scale)
        logger.info(f"Using manual scale: {scale}")
    else:
        if detect_method == "runs":
            scale = (
                runs_based_detect_accelerated(current)
                if "RUST_AVAILABLE" in globals() and RUST_AVAILABLE
                else runs_based_detect(current)
            )
        elif detect_method == "edge":
            scale = edge_aware_detect(current)
        else:  # auto
            scale = (
                runs_based_detect_accelerated(current)
                if "RUST_AVAILABLE" in globals() and RUST_AVAILABLE
                else runs_based_detect(current)
            )
            if scale <= 1:
                logger.info("Runs detection failed, trying edge detection")
                scale = edge_aware_detect(current)

    # 2.5. Snap to grid
    if snap_grid and scale > 1:
        logger.info("Snapping to grid...")
        gray = cv2.cvtColor(current, cv2.COLOR_RGBA2GRAY)
        crop_x, crop_y = find_optimal_crop(gray, scale)

        # Calculate new size divisible by scale
        new_width = ((current.shape[1] - crop_x) // scale) * scale
        new_height = ((current.shape[0] - crop_y) // scale) * scale

        if new_width >= scale and new_height >= scale:
            current = current[crop_y : crop_y + new_height, crop_x : crop_x + new_width]
            logger.info(f"Cropped to {new_width}x{new_height} from offset ({crop_x}, {crop_y})")

    # 3. Optional cleanup
    if cleanup.get("morph", False):
        current = morphological_cleanup(current)

    # 3.5. Auto-detect optimal colors if requested
    if max_colors is None or auto_color_detect:
        max_colors = detect_optimal_color_count(current)
        logger.info(f"Auto-detected optimal color count: {max_colors}")
    else:
        max_colors = max_colors or 32  # Default if not specified

    # 4. Color Quantization
    initial_colors = count_colors(current)
    colors_used = initial_colors

    if max_colors < 256 and initial_colors > max_colors:
        logger.info(f"Quantizing from {initial_colors} to max {max_colors} colors")
        current, palette_colors = quantize_colors(current, max_colors, fixed_palette)
        colors_used = len(palette_colors)

    # Iteration loop for rerun
    best_current = current.copy()
    best_score = float('inf')  # Lower is better: colors + (1 - normalized_variance)

    for iter in range(iterations):
        iter_threshold = dom_mean_threshold + (iter - iterations//2) * 0.01  # Vary slightly
        current_iter = current.copy()

        # Pre-filter if enabled
        if pre_filter:
            current_iter = pre_downscale_filter(current_iter)

        # 5. Downscaling
        if scale > 1 or downscale_method in ["content-adaptive", "hybrid"]:
            logger.info(f'Downscaling by {scale}x using "{downscale_method}" method')
            if downscale_method == "dominant":
                # Use accelerated version if available
                if "RUST_AVAILABLE" in globals() and RUST_AVAILABLE:
                    current_iter = downscale_dominant_color_accelerated(current_iter, scale, iter_threshold)
                else:
                    current_iter = downscale_by_dominant_color(current_iter, scale, iter_threshold)  
            elif downscale_method == "content-adaptive":
                target_w = current_iter.shape[1] // scale if scale > 1 else current_iter.shape[1] // 2
                target_h = current_iter.shape[0] // scale if scale > 1 else current_iter.shape[0] // 2
                current_iter = content_adaptive_downscale(current_iter, target_w, target_h)
                # Post-quantize after content-adaptive
                if max_colors < 256:
                    current_iter, _ = quantize_colors(current_iter, max_colors, fixed_palette)
                if edge_preserve:
                    current_iter = edge_preserving_refinement(current_iter)
            elif downscale_method == "hybrid":
                current_iter = hybrid_downscale(current_iter, scale, iter_threshold)
            else:
                current_iter = downscale_block(current_iter, scale, downscale_method)
            current_iter = finalize_pixels(current_iter)

        # Post-sharpen if enabled
        if post_sharpen:
            current_iter = post_downscale_sharpen(current_iter)

        # 6. Post-cleanup
        if cleanup.get("jaggy", False):
            current_iter = jaggy_cleaner(current_iter)

        # Score: lower colors + higher edges (normalized)
        final_colors = count_colors(current_iter)
        gray = cv2.cvtColor(current_iter, cv2.COLOR_RGBA2GRAY if current_iter.shape[2] == 4 else cv2.COLOR_RGB2GRAY)
        edge_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        norm_var = edge_var / (gray.shape[0] * gray.shape[1])  # Normalize
        score = final_colors - norm_var  # Minimize colors, maximize var

        if score < best_score:
            best_score = score
            best_current = current_iter

    current = best_current

    # 7. Extract palette and save
    palette = extract_palette(current)

    # Convert back to PIL and save as PNG
    output_image = Image.fromarray(current)

    processing_time = int((time.time() - start_time) * 1000)

    # Generate manifest
    manifest = ProcessingManifest(
        original_size=original_size,
        final_size=(current.shape[1], current.shape[0]),
        processing_steps={
            "scale_detection": {
                "method": detect_method,
                "detected_scale": scale,
                "manual_scale": manual_scale,
            },
            "color_quantization": {
                "max_colors": max_colors,
                "initial_colors": initial_colors,
                "final_colors": colors_used,
                "fixed_palette": len(fixed_palette) if fixed_palette else None,
            },
            "downscaling": {
                "method": downscale_method,
                "scale_factor": scale,
                "dom_mean_threshold": dom_mean_threshold,
                "applied": scale > 1,
            },
            "cleanup": cleanup,
            "alpha_processing": {
                "threshold": alpha_threshold,
                "binarized": alpha_threshold is not None,
            },
            "grid_snapping": {"enabled": snap_grid, "applied": snap_grid and scale > 1},
            "pre_filter": pre_filter,
            "edge_preserve": edge_preserve,
            "post_sharpen": post_sharpen,
            "iterations": iterations,
        },
        processing_time_ms=processing_time,
        timestamp=datetime.now().isoformat(),
    )

    logger.info(f"Processing complete in {processing_time}ms")

    return {"image": output_image, "image_array": current, "palette": palette, "manifest": manifest}


# Synchronous wrapper for the async function
def process_image_sync(file_path: str, **kwargs: Any) -> Dict:
    """Synchronous version of process_image"""
    import asyncio

    return asyncio.run(process_image(file_path, **kwargs))


def main() -> None:
    """Main entry point for the command-line interface"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Advanced pixel art optimization tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pixel-art input.png                          # Basic processing
  pixel-art input.png -o output.png            # Specify output file
  pixel-art input.png -c 16 -m dominant        # 16 colors, dominant downscaling
  pixel-art input.png --auto-colors            # Auto-detect color count
  pixel-art input.png --scale 4                # Force 4x downscaling
  pixel-art input.png --cleanup morph,jaggy    # Enable cleanup options
        """,
    )

    parser.add_argument("input", help="Input image file")
    parser.add_argument(
        "-o", "--output", default="output.png", help="Output image file (default: output.png)"
    )
    parser.add_argument(
        "-c", "--colors", type=int, help="Maximum number of colors (default: auto-detect)"
    )
    parser.add_argument(
        "--auto-colors", action="store_true", help="Auto-detect optimal color count"
    )
    parser.add_argument("-s", "--scale", type=int, help="Manual scale override")
    parser.add_argument(
        "-d",
        "--detect",
        choices=["auto", "runs", "edge"],
        default="auto",
        help="Scale detection method (default: auto)",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["dominant", "median", "mode", "mean", "nearest", "content-adaptive", "hybrid"],
        default="dominant",
        help="Downscaling method (default: dominant)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05, help="Dominant color threshold (default: 0.05)"
    )
    parser.add_argument("--cleanup", help="Cleanup options: morph,jaggy (comma-separated)")
    parser.add_argument("--palette", help="Fixed palette file (hex colors, one per line)")
    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=128,
        help="Alpha binarization threshold (default: 128)",
    )
    parser.add_argument("--no-snap", action="store_true", help="Disable grid snapping")
    parser.add_argument("--pre-filter", action="store_true", help="Apply pre-downscale filter")
    parser.add_argument("--edge-preserve", action="store_true", help="Apply edge-preserving refinement")
    parser.add_argument("--post-sharpen", action="store_true", help="Apply post-downscale sharpening")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations (default: 1)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse cleanup options
    cleanup_options = {"morph": False, "jaggy": False}
    if args.cleanup:
        for option in args.cleanup.split(","):
            option = option.strip().lower()
            if option in cleanup_options:
                cleanup_options[option] = True
            else:
                print(f"Warning: Unknown cleanup option '{option}'")

    # Load fixed palette if specified
    fixed_palette = None
    if args.palette:
        try:
            with open(args.palette) as f:
                fixed_palette = [line.strip() for line in f if line.strip().startswith("#")]
        except FileNotFoundError:
            print(f"Error: Palette file '{args.palette}' not found")
            sys.exit(1)

    try:
        result = process_image_sync(
            args.input,
            max_colors=args.colors,
            manual_scale=args.scale,
            detect_method=args.detect,
            downscale_method=args.method,
            dom_mean_threshold=args.threshold,
            cleanup=cleanup_options,
            fixed_palette=fixed_palette,
            alpha_threshold=args.alpha_threshold,
            snap_grid=not args.no_snap,
            auto_color_detect=args.auto_colors,
            pre_filter=args.pre_filter,
            edge_preserve=args.edge_preserve,
            post_sharpen=args.post_sharpen,
            iterations=args.iterations,
        )

        # Save the processed image
        result["image"].save(args.output)

        if not args.quiet:
            # Print summary
            manifest = result["manifest"]
            print(f"✓ Processing complete!")
            print(
                f"  Input: {args.input} ({manifest.original_size[0]}x{manifest.original_size[1]})"
            )
            print(f"  Output: {args.output} ({manifest.final_size[0]}x{manifest.final_size[1]})")
            print(f"  Colors: {manifest.processing_steps['color_quantization']['final_colors']}")

            if manifest.processing_steps["scale_detection"]["detected_scale"] > 1:
                print(
                    f"  Scale: {manifest.processing_steps['scale_detection']['detected_scale']}x detected"
                )

            print(f"  Time: {manifest.processing_time_ms}ms")

            if args.verbose:
                print(f"\nPalette ({len(result['palette'])} colors):")
                for i, color in enumerate(result["palette"]):
                    if i % 8 == 0 and i > 0:
                        print()
                    print(color, end=" ")
                print()

                print("\nProcessing Manifest:")
                print(json.dumps(manifest.__dict__, indent=2))

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()