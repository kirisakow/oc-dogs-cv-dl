"""
Example usage of image processing functions for CNN preprocessing.
"""

import numpy as np
from functions import (
    preprocess_for_cnn,
    resize_image,
    equalize_histogram,
    whiten_image
)


def example_cnn_preprocessing_pipeline():
    """
    Example of how to use the preprocessing functions for CNN training.
    """
    print("CNN Image Preprocessing Pipeline Example")
    print("=" * 50)

    # Simulate loading an image (in practice, use cv2.imread())
    # For this example, we'll create a random image
    sample_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    print(f"Original image shape: {sample_image.shape}")

    # Example 1: Basic preprocessing (resize + normalize)
    print("\n1. Basic preprocessing (resize + normalize):")
    preprocessed_basic = preprocess_for_cnn(
        sample_image,
        target_size=(224, 224),  # Common size for CNNs
        whiten=False,
        equalize=False,
        normalize=True
    )
    print(f"   Result shape: {preprocessed_basic.shape}, dtype: {preprocessed_basic.dtype}")
    print(f"   Pixel range: [{np.min(preprocessed_basic):.3f}, {np.max(preprocessed_basic):.3f}]")

    # Example 2: Enhanced preprocessing (resize + equalize + normalize)
    print("\n2. Enhanced preprocessing (resize + equalize + normalize):")
    preprocessed_enhanced = preprocess_for_cnn(
        sample_image,
        target_size=(224, 224),
        whiten=False,
        equalize=True,  # Apply histogram equalization
        normalize=True
    )
    print(f"   Result shape: {preprocessed_enhanced.shape}, dtype: {preprocessed_enhanced.dtype}")

    # Example 3: Advanced preprocessing (resize + equalize + whiten + normalize)
    print("\n3. Advanced preprocessing (resize + equalize + whiten + normalize):")
    preprocessed_advanced = preprocess_for_cnn(
        sample_image,
        target_size=(224, 224),
        whiten=True,     # Apply ZCA whitening
        equalize=True,   # Apply histogram equalization
        normalize=True    # Normalize pixel values
    )
    print(f"   Result shape: {preprocessed_advanced.shape}, dtype: {preprocessed_advanced.dtype}")

    # Example 4: Custom preprocessing pipeline
    print("\n4. Custom preprocessing pipeline:")

    # Step 1: Resize
    resized = resize_image(sample_image, (256, 256))
    print(f"   After resize: {resized.shape}")

    # Step 2: Equalize histogram
    equalized = equalize_histogram(resized, clip_limit=2.0, grid_size=(8, 8))
    print(f"   After equalization: {equalized.shape}")

    # Step 3: Apply whitening
    whitened = whiten_image(equalized)
    print(f"   After whitening: {whitened.shape}")

    # Step 4: Final resize to target CNN input size
    final = resize_image(whitened, (224, 224))
    print(f"   Final size: {final.shape}")

    print("\n" + "=" * 50)
    print("Preprocessing complete! Images are ready for CNN training.")


def example_batch_preprocessing():
    """
    Example of batch preprocessing for multiple images.
    """
    print("\nBatch Preprocessing Example")
    print("=" * 30)

    # Simulate a batch of images
    batch_size = 4
    batch_images = [np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8) for _ in range(batch_size)]

    print(f"Processing batch of {batch_size} images...")

    # Process each image in the batch
    preprocessed_batch = []
    for i, img in enumerate(batch_images):
        processed = preprocess_for_cnn(
            img,
            target_size=(128, 128),
            equalize=True,
            normalize=True
        )
        preprocessed_batch.append(processed)
        print(f"  Image {i + 1}: {img.shape} -> {processed.shape}")

    # Convert to numpy array for CNN input
    batch_array = np.array(preprocessed_batch)
    print(f"\nFinal batch shape: {batch_array.shape}")
    print(f"Batch dtype: {batch_array.dtype}")

    print("Batch preprocessing complete!")


if __name__ == "__main__":
    example_cnn_preprocessing_pipeline()
    example_batch_preprocessing()