# unfake-opt

Improve AI-generated pixel art through scale detection, color quantization, and smart downscaling — now significantly faster and more accurate thanks to algorithmic and performance enhancements.  
This optimized fork features **10–40× faster content-adaptive downscaling**, improved dominant color selection using **KMeans**, a new **hybrid downscaling method**, and additional preprocessing/postprocessing options for sharper, cleaner pixel art.

Based on the excellent work by:
- **Eugeniy Smirnov** ([jenissimo/unfake.js](https://github.com/jenissimo/unfake.js)) – Original JavaScript implementation  
- **Igor Bezkrovnyi** ([ibezkrovnyi/image-quantization](https://github.com/ibezkrovnyi/image-quantization)) – Image quantization algorithms  
- **Benjamin Paine** ([painebenjamin/unfake.py](https://github.com/painebenjamin/unfake.py)) – Original Python/Rust port  

## Examples  

Original Generated Image
![Original Generated Image](https://raw.githubusercontent.com/2dameneko/unfake-opt.py/main/samples/orig_2025-09-08-005621__0.png)

Original Dominant color method
![Original Dominant color method](https://raw.githubusercontent.com/2dameneko/unfake-opt.py/main/samples/orig_dom_pixelart_2025-09-08-005621__0_8x.png)

Enhanced Dominant color method
![Enhanced Dominant color method](https://raw.githubusercontent.com/2dameneko/unfake-opt.py/main/samples/enh_dom_pixelart_2025-09-08-005621__0_8x.png)

---

## ✨ Key Improvements (vs. original port)

- **10–40× faster `content-adaptive` downscaling** via optimized Rust implementation  
- **Improved `dominant` method**: uses **KMeans clustering** for better color selection, especially on complex pixel-art backgrounds  
- **New `hybrid` downscaling method**: automatically combine the best from `dominant` and `content-adaptive` methods
- **Preprocessing**: optional light blur (`--pre-filter`) before quantization to reduce noise  
- **Edge preservation**: `--edge-preserve` enhances contour sharpness during downscaling  
- **Post-sharpening**: experimental `--post-sharpen` (currently under refinement, produce mostly unwanted results)  
- **Adaptive threshold tuning**: `--iterations N` allows iterative refinement of the dominant color threshold for `dominant` method  

---

## Features

- **Automatic Scale Detection**: Detects the inherent scale of pixel art using both runs-based and edge-aware methods  
- **Advanced Color Quantization**: Wu algorithm with Rust acceleration + KMeans-enhanced dominant color selection  
- **Smart Downscaling**: Multiple methods including `dominant`, `median`, `mode`, `content-adaptive`, and new `hybrid`  
- **Image Cleanup**: Alpha binarization, morphological operations, jaggy edge removal  
- **Grid Snapping**: Automatic alignment to pixel grid for clean results  
- **Flexible API**: Both synchronous and asynchronous interfaces  
- **Blazing Fast**: Process a 1-megapixel image in under a second (with Rust acceleration)

### Upcoming

- Refined post-sharpening algorithm  
- Vectorization support  

---

## Installation

### From Source (recommended for now)

```bash
# Clone the optimized fork
git clone https://github.com/2dameneko/unfake-opt.py.git
cd unfake-opt

# Install with pip (includes Rust compilation)
pip install .

# Or for development
pip install -e .
```

### From precompiled wheel (after release)

> **Note**: This fork is not yet published on PyPI. Install from source to access all new features.

### Requirements

- Python 3.8+  
- Rust toolchain (for building from source)  
- OpenCV Python bindings  
- Pillow  
- NumPy  
- scikit-learn (for KMeans in `dominant` method)

---

## Usage

### Command Line

```bash
# Basic usage with auto-detection
unfake input.png

# Specify output file
unfake input.png -o output.png

# Control color palette size
unfake input.png -c 16                    # Maximum 16 colors
unfake input.png --auto-colors            # Auto-detect optimal color count

# Force specific scale
unfake input.png --scale 4                # Force 4x downscaling

# Choose downscaling method (NEW: hybrid!)
unfake input.png -m dominant              # Dominant color (KMeans-enhanced, default)
unfake input.png -m content-adaptive      # High-quality, now 10–40× faster
unfake input.png -m hybrid                # NEW: best of dominant + content-adaptive

# Enable new preprocessing/postprocessing
unfake input.png --pre-filter             # Apply light blur before quantization
unfake input.png --edge-preserve          # Preserve sharp edges during downscaling
unfake input.png --post-sharpen           # Experimental sharpening after quantization, not recommended for now
unfake input.png --iterations 5           # Refine dominant threshold over 5 iterations

# Enable cleanup operations
unfake input.png --cleanup morph,jaggy    # Morphological + jaggy edge cleanup

# Use fixed color palette
unfake input.png --palette palette.txt    # File with hex colors, one per line

# Adjust processing parameters
unfake input.png --alpha-threshold 200    # Higher threshold for alpha binarization
unfake input.png --threshold 0.1          # Initial dominant color threshold (0.0–1.0)
unfake input.png --no-snap                # Disable grid snapping

# Verbose output
unfake input.png -v                       # Show detailed processing info
```

### Python API

```python
import unfake

# Basic processing with defaults (now uses KMeans-enhanced dominant)
result = unfake.process_image_sync(
    "input.png",
    max_colors=32,
    detect_method="auto",
    downscale_method="hybrid",            # NEW option!
    cleanup={"morph": False, "jaggy": False},
    snap_grid=True,
    pre_filter=True,                      # NEW
    edge_preserve=True,                   # NEW
    post_sharpen=False,                   # Experimental
    iterations=3                          # NEW: threshold refinement
)

# Access results
processed_image = result['image']        # PIL Image
palette = result['palette']              # List of hex colors
manifest = result['manifest']            # Processing metadata
```

#### Asynchronous API (unchanged, but faster)

```python
import asyncio
import unfake

async def process_image_async():
    result = await unfake.process_image(
        "input.png",
        max_colors=16,
        downscale_method="hybrid",
        pre_filter=True,
        edge_preserve=True
    )
    result["image"].save("output.png")

asyncio.run(process_image_async())
```

---

### New & Updated Processing Options

#### Downscaling Methods
- **`dominant`** (default): Now uses **KMeans clustering** for more accurate dominant color selection — especially effective on textured or gradient pixel-art backgrounds  
- **`content-adaptive`**: Same high-quality algorithm, but **10–40× faster** thanks to Rust optimization  
- **`hybrid`** (**NEW**): Combine best from `dominant` and `content-adaptive` for optimal fidelity  
- **`median` / `mode` / `mean`**: Unchanged, for compatibility

#### New Flags
- `--pre-filter`: Applies a slight Gaussian blur before quantization to reduce noise and improve color coherence  
- `--edge-preserve`: Enhances edge contrast during downscaling to maintain crisp silhouettes  
- `--post-sharpen`: Experimental unsharp masking after quantization (not recommended for now)  
- `--iterations N`: Runs N iterations of threshold tuning for the `dominant` method to find optimal color dominance cutoff  

---

## Algorithm Details

### Dominant Color (Enhanced)
- Uses **KMeans clustering** in RGB space to group similar colors  
- Selects the cluster centroid with the most pixels as the representative color  
- Better handles dithering, gradients, and noisy backgrounds common in AI-generated pixel art  

### Hybrid Downscaling
- For each scale×scale block:
  - Compute results from both `dominant` and `content-adaptive`
  - Combine results based of details frequency (low - dominant, high - adaptive)

### Content Adaptive Downscaling
- Roughly O(num_kernels * num_pixels) => O(num_pixels) per iteration

---

## Credits

This optimized fork builds upon:

- **[unfake.js](https://github.com/jenissimo/unfake.js)** by Eugeniy Smirnov  
- **[image-quantization](https://github.com/ibezkrovnyi/image-quantization)** by Igor Bezkrovnyi  
- **[unfake.py](https://github.com/painebenjamin/unfake.py)** by Benjamin Paine  

Additional references:  
- Wu, Xiaolin. "Efficient Statistical Computations for Optimal Color Quantization" (1992)  
- Kopf, Johannes and Dani Lischinski. "Depixelizing Pixel Art" (2011)  
- Scikit-learn: KMeans implementation for color clustering  

---

## License

MIT License
