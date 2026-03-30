"""
Image processing functions for CNN preprocessing pipeline.
Includes whitening, equalization, scaling, and other preprocessing utilities.
"""

from lxml import etree
from typing import Union, Tuple
import cv2
import numpy as np


def get_boundingbox(path):
    annotation = etree.parse(path)
    return tuple(int(annotation.xpath(f"/annotation/object/bndbox/{coord}").pop().text) for coord in ['xmin', 'ymin', 'xmax', 'ymax'])


def get_breed(path):
    annotation = etree.parse(path)
    return annotation.xpath("/annotation/object/name").pop().text


def whiten_image(image: np.ndarray) -> np.ndarray:
    """
    Perform image whitening (ZCA whitening) for normalization.

    Args:
        image: Input image as numpy array (H, W, C)

    Returns:
        Whitened image as numpy array
    """
    # Convert to float and reshape for covariance calculation
    img_float = image.astype(np.float32) / 255.0
    h, w, c = img_float.shape
    img_reshaped = img_float.reshape(h * w, c)

    # Center the data
    img_centered = img_reshaped - np.mean(img_reshaped, axis=0)

    # Calculate covariance matrix
    covariance = np.cov(img_centered, rowvar=False)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-5

    # Perform ZCA whitening
    U, S, V = np.linalg.svd(covariance)
    whitening_matrix = U @ np.diag(1.0 / np.sqrt(S + epsilon)) @ U.T

    # Apply whitening
    whitened = img_centered @ whitening_matrix.T

    # Reshape back to original dimensions
    whitened_img = whitened.reshape(h, w, c)

    # Scale back to 0-255 range
    whitened_img = (whitened_img - np.min(whitened_img)) / (np.max(whitened_img) - np.min(whitened_img)) * 255.0

    return whitened_img.astype(np.uint8)


def equalize_histogram(image: np.ndarray, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image contrast.

    Args:
        image: Input image as numpy array
        clip_limit: Threshold for contrast limiting (default: 2.0)
        grid_size: Grid size for CLAHE (default: (8, 8))

    Returns:
        Equalized image as numpy array
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to YUV color space for better results on color images
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

        # Apply CLAHE to the Y channel (luminance)
        yuv_image[:, :, 0] = clahe.apply(yuv_image[:, :, 0])

        # Convert back to BGR
        equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        equalized_image = clahe.apply(image)

    return equalized_image


def resize_image(image: np.ndarray, target_size: Tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize image to target dimensions.

    Args:
        image: Input image as numpy array
        target_size: Target size as (width, height) tuple
        interpolation: OpenCV interpolation method (default: cv2.INTER_LINEAR)

    Returns:
        Resized image as numpy array
    """
    return cv2.resize(image, target_size, interpolation=interpolation)


def normalize_image(image: np.ndarray, mean: Union[float, Tuple[float, float, float]] = 0.0,
                    std: Union[float, Tuple[float, float, float]] = 1.0) -> np.ndarray:
    """
    Normalize image pixel values.

    Args:
        image: Input image as numpy array
        mean: Mean value(s) to subtract (default: 0.0)
        std: Standard deviation value(s) to divide by (default: 1.0)

    Returns:
        Normalized image as numpy array (float32)
    """
    image_normalized = image.astype(np.float32) / 255.0

    if isinstance(mean, (tuple, list)):
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)

        # Subtract mean and divide by std for each channel
        for i in range(image.shape[2]):
            image_normalized[:, :, i] = (image_normalized[:, :, i] - mean[i]) / std[i]
    else:
        image_normalized = (image_normalized - mean) / std

    return image_normalized


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert color image to grayscale.

    Args:
        image: Input image as numpy array

    Returns:
        Grayscale image as numpy array
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5),
                       sigma_x: float = 0.0) -> np.ndarray:
    """
    Apply Gaussian blur for noise reduction.

    Args:
        image: Input image as numpy array
        kernel_size: Size of the Gaussian kernel (default: (5, 5))
        sigma_x: Standard deviation in X direction (default: 0.0 - auto-calculated)

    Returns:
        Blurred image as numpy array
    """
    return cv2.GaussianBlur(image, kernel_size, sigmaX=sigma_x)


def preprocess_for_cnn(image: np.ndarray, target_size: Tuple[int, int] = (224, 224),
                      whiten: bool = False, equalize: bool = False,
                      normalize: bool = True) -> np.ndarray:
    """
    Complete preprocessing pipeline for CNN input.

    Args:
        image: Input image as numpy array
        target_size: Target size for resizing (default: (224, 224))
        whiten: Apply whitening if True (default: False)
        equalize: Apply histogram equalization if True (default: False)
        normalize: Normalize pixel values if True (default: True)

    Returns:
        Preprocessed image ready for CNN input
    """
    # Convert to color if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Resize
    image = resize_image(image, target_size)

    # Equalize histogram if requested
    if equalize:
        image = equalize_histogram(image)

    # Whiten if requested
    if whiten:
        image = whiten_image(image)

    # Normalize
    if normalize:
        image = normalize_image(image)

    return image