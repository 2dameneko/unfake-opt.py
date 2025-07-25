import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import Rust acceleration
try:
    from unfake.unfake import WuQuantizerRust
    from unfake.unfake import count_unique_colors as count_colors_rust
    from unfake.unfake import downscale_dominant_color as downscale_dominant_rust
    from unfake.unfake import finalize_pixels_rust
    from unfake.unfake import map_pixels_to_palette as map_pixels_to_palette_rust
    from unfake.unfake import runs_based_detect as runs_based_detect_rust

    RUST_AVAILABLE = True
    logger.info("Rust acceleration is available")
except ImportError:
    RUST_AVAILABLE = False
    logger.warning("Rust acceleration not available, using Python implementations")


def runs_based_detect_accelerated(image: np.ndarray) -> int:
    """
    Detect scale using runs-based method with Rust acceleration if available.

    Args:
        image: Input image as numpy array (H, W, C)

    Returns:
        Detected scale factor
    """
    if RUST_AVAILABLE:
        return runs_based_detect_rust(image)
    else:
        # Fall back to Python implementation
        from pixel import runs_based_detect

        return runs_based_detect(image)


def map_pixels_to_palette_accelerated(
    pixels: np.ndarray, palette: List[Tuple[int, int, int]]
) -> np.ndarray:
    """
    Map pixels to nearest palette colors with Rust acceleration if available.

    Args:
        pixels: Input image as numpy array (H, W, C)
        palette: List of RGB tuples

    Returns:
        Quantized image
    """
    if RUST_AVAILABLE:
        return map_pixels_to_palette_rust(pixels, palette)
    else:
        # Fall back to Python implementation
        h, w, c = pixels.shape
        has_alpha = c == 4
        quantized = np.zeros_like(pixels)

        for y in range(h):
            for x in range(w):
                if has_alpha and pixels[y, x, 3] < 128:
                    quantized[y, x] = [0, 0, 0, 0]
                else:
                    pixel = pixels[y, x, :3]
                    min_dist = float("inf")
                    best_color = palette[0]

                    for color in palette:
                        dist = np.sum((pixel.astype(int) - np.array(color).astype(int)) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            best_color = color

                    quantized[y, x, 0] = best_color[0]
                    quantized[y, x, 1] = best_color[1]
                    quantized[y, x, 2] = best_color[2]
                    if has_alpha:
                        quantized[y, x, 3] = 255

        return quantized.astype(np.uint8)


def downscale_dominant_color_accelerated(
    image: np.ndarray, scale: int, threshold: float = 0.05
) -> np.ndarray:
    """
    Downscale using dominant color method with Rust acceleration if available.

    Args:
        image: Input image as numpy array (H, W, C)
        scale: Scale factor for downscaling
        threshold: Threshold for dominant color vs mean (0.0-1.0)

    Returns:
        Downscaled image
    """
    if RUST_AVAILABLE:
        return downscale_dominant_rust(image, scale, threshold)
    else:
        # Fall back to Python implementation
        from pixel import downscale_by_dominant_color

        return downscale_by_dominant_color(image, scale, threshold)


def count_colors_accelerated(image: np.ndarray) -> int:
    """
    Count unique opaque colors with Rust acceleration if available.

    Args:
        image: Input image as numpy array (H, W, C)

    Returns:
        Number of unique colors
    """
    if RUST_AVAILABLE:
        return count_colors_rust(image)
    else:
        # Fall back to Python implementation
        from pixel import count_colors

        return count_colors(image)


def finalize_pixels_accelerated(image: np.ndarray) -> np.ndarray:
    """
    Finalize pixels (binary alpha, black transparent) with Rust acceleration if available.

    Args:
        image: Input image as numpy array (H, W, C)

    Returns:
        Finalized image
    """
    if RUST_AVAILABLE:
        return finalize_pixels_rust(image)
    else:
        # Fall back to Python implementation
        from pixel import finalize_pixels

        return finalize_pixels(image)


class WuQuantizerAccelerated:
    """
    Wu color quantizer with Rust acceleration if available.
    """

    def __init__(self, max_colors: int = 256, significant_bits: int = 5):
        self.max_colors = max_colors
        self.significant_bits = significant_bits

        if RUST_AVAILABLE:
            self._impl = WuQuantizerRust(max_colors, significant_bits)
            self._use_rust = True
        else:
            # Fall back to Python implementation
            from wu_quantizer import WuQuantizer

            self._impl = WuQuantizer(max_colors, significant_bits)
            self._use_rust = False

    def quantize(self, pixels: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """
        Quantize image colors using Wu algorithm.

        Args:
            pixels: Input image as numpy array (H, W, C)

        Returns:
            Tuple of (quantized_image, palette)
        """
        if self._use_rust:
            # Rust version returns the same format
            return self._impl.quantize(pixels)
        else:
            # Python version compatibility
            return self._impl.quantize(pixels)


# Export the accelerated versions
__all__ = [
    "runs_based_detect_accelerated",
    "map_pixels_to_palette_accelerated",
    "downscale_dominant_color_accelerated",
    "count_colors_accelerated",
    "finalize_pixels_accelerated",
    "WuQuantizerAccelerated",
    "RUST_AVAILABLE",
]
