# Image Processing Functions for CNN Preprocessing

This module provides comprehensive image processing functions for preprocessing images before CNN training.

## Available Functions

### Core Processing Functions

1. **`whiten_image(image: np.ndarray) -> np.ndarray`**
   - Performs ZCA whitening for image normalization
   - Reduces redundancy between color channels
   - Input: RGB image as numpy array (H, W, C)
   - Output: Whitened image as numpy array

2. **`equalize_histogram(image: np.ndarray, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray`**
   - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Enhances local contrast while limiting noise amplification
   - Works on both color and grayscale images
   - `clip_limit`: Contrast threshold (default: 2.0)
   - `grid_size`: Tile grid size for local processing (default: (8, 8))

3. **`resize_image(image: np.ndarray, target_size: Tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray`**
   - Resizes image to target dimensions
   - `target_size`: (width, height) tuple
   - `interpolation`: OpenCV interpolation method

4. **`normalize_image(image: np.ndarray, mean: Union[float, Tuple[float, float, float]] = 0.0, std: Union[float, Tuple[float, float, float]] = 1.0) -> np.ndarray`**
   - Normalizes pixel values to [0, 1] range and applies mean/std normalization
   - Supports both single value and per-channel normalization
   - Output is float32 dtype

### Utility Functions

5. **`convert_to_grayscale(image: np.ndarray) -> np.ndarray`**
   - Converts color image to grayscale
   - Preserves grayscale images unchanged

6. **`apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), sigma_x: float = 0.0) -> np.ndarray`**
   - Applies Gaussian blur for noise reduction
   - `kernel_size`: Size of the Gaussian kernel
   - `sigma_x`: Standard deviation in X direction

### Complete Preprocessing Pipeline

7. **`preprocess_for_cnn(image: np.ndarray, target_size: Tuple[int, int] = (224, 224), whiten: bool = False, equalize: bool = False, normalize: bool = True) -> np.ndarray`**
   - Complete preprocessing pipeline for CNN input
   - Handles grayscale to color conversion automatically
   - Parameters:
     - `target_size`: Final image size (default: (224, 224))
     - `whiten`: Apply ZCA whitening if True
     - `equalize`: Apply histogram equalization if True
     - `normalize`: Normalize pixel values if True
   - Returns preprocessed image ready for CNN training

## Usage Examples

### Basic Usage
```python
import cv2
from functions import preprocess_for_cnn

# Load image
image = cv2.imread("path/to/image.jpg")

# Basic preprocessing (resize + normalize)
processed = preprocess_for_cnn(image, target_size=(224, 224))

# Enhanced preprocessing (resize + equalize + normalize)
processed = preprocess_for_cnn(image, target_size=(224, 224), equalize=True)

# Advanced preprocessing (resize + equalize + whiten + normalize)
processed = preprocess_for_cnn(image, target_size=(224, 224), 
                               equalize=True, whiten=True)
```

### Batch Processing
```python
import numpy as np
from functions import preprocess_for_cnn

# List of images (numpy arrays)
batch_images = [img1, img2, img3, img4]

# Process batch
preprocessed_batch = []
for img in batch_images:
    processed = preprocess_for_cnn(img, target_size=(128, 128), equalize=True)
    preprocessed_batch.append(processed)

# Convert to numpy array for CNN input
batch_array = np.array(preprocessed_batch)  # Shape: (batch_size, height, width, channels)
```

### Custom Pipeline
```python
from functions import resize_image, equalize_histogram, whiten_image, normalize_image

# Step-by-step preprocessing
resized = resize_image(original_image, (256, 256))
equalized = equalize_histogram(resized, clip_limit=2.0)
whitened = whiten_image(equalized)
final = resize_image(whitened, (224, 224))
normalized = normalize_image(final)
```

## Dependencies

- OpenCV (cv2)
- NumPy
- Python 3.12+

## Notes

- All functions work with numpy arrays in OpenCV format (BGR color order)
- For PIL Image objects, convert using `np.array(image)` or `cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)`
- The preprocessing functions are designed to be modular and can be used individually or combined
- Normalization outputs float32 arrays suitable for most deep learning frameworks