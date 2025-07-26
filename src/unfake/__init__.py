"""
unfake - AI-generated pixel art optimization tool

A high-performance tool for improving AI-generated pixel art through scale detection,
color quantization, and smart downscaling. Features Rust acceleration for critical
operations.
"""

__version__ = "1.0.0"

# Import main functions from pixel module
from .pixel import (
    main,
    process_image,
    process_image_sync,
    runs_based_detect,
    edge_aware_detect,
    quantize_colors,
    extract_palette,
    count_colors,
    alpha_binarization,
    morphological_cleanup,
    jaggy_cleaner,
    finalize_pixels,
    detect_optimal_color_count,
    downscale_by_dominant_color,
    downscale_block
)

# Import content-adaptive downscaling
from .content_adaptive import content_adaptive_downscale

# Import accelerated versions from rust integration
from .pixel_rust_integration import (
    runs_based_detect_accelerated,
    map_pixels_to_palette_accelerated,
    WuQuantizerAccelerated,
    RUST_AVAILABLE
)

__all__ = [
    # Main processing functions
    'process_image',
    'process_image_sync',
    'main',
    
    # Core algorithms
    'runs_based_detect',
    'edge_aware_detect',
    'quantize_colors',
    'extract_palette',
    'count_colors',
    'detect_optimal_color_count',
    
    # Downscaling methods
    'downscale_by_dominant_color',
    'downscale_block',
    
    # Image cleanup
    'alpha_binarization',
    'morphological_cleanup',
    'jaggy_cleaner',
    'finalize_pixels',
    
    # Advanced features
    'content_adaptive_downscale',
    
    # Rust-accelerated functions
    'runs_based_detect_accelerated',
    'map_pixels_to_palette_accelerated',
    'WuQuantizerAccelerated',
    'RUST_AVAILABLE',
    
    # Version
    '__version__'
] 
