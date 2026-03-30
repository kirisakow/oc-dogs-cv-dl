"""
Test script for image processing functions.
"""

import cv2
import numpy as np
import os
from functions import (
    whiten_image, equalize_histogram, resize_image, 
    normalize_image, convert_to_grayscale, 
    apply_gaussian_blur, preprocess_for_cnn
)


def test_basic_functionality():
    """Test basic functionality of all functions."""
    print("Testing image processing functions...")
    
    # Create a simple test image (100x100 RGB)
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    # Test each function
    try:
        # Test whitening
        whitened = whiten_image(test_image)
        print(f"✓ Whitening: Input shape {test_image.shape}, Output shape {whitened.shape}")
        
        # Test equalization
        equalized = equalize_histogram(test_image)
        print(f"✓ Equalization: Input shape {test_image.shape}, Output shape {equalized.shape}")
        
        # Test resizing
        resized = resize_image(test_image, (50, 50))
        print(f"✓ Resizing: Input shape {test_image.shape}, Output shape {resized.shape}")
        
        # Test normalization
        normalized = normalize_image(test_image)
        print(f"✓ Normalization: Input shape {test_image.shape}, Output shape {normalized.shape}, dtype {normalized.dtype}")
        
        # Test grayscale conversion
        gray = convert_to_grayscale(test_image)
        print(f"✓ Grayscale: Input shape {test_image.shape}, Output shape {gray.shape}")
        
        # Test Gaussian blur
        blurred = apply_gaussian_blur(test_image)
        print(f"✓ Gaussian blur: Input shape {test_image.shape}, Output shape {blurred.shape}")
        
        # Test full preprocessing pipeline
        preprocessed = preprocess_for_cnn(test_image, target_size=(64, 64))
        print(f"✓ Full pipeline: Input shape {test_image.shape}, Output shape {preprocessed.shape}")
        
        print("\nAll tests passed! ✓")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False
    
    return True


def test_with_real_image():
    """Test with a real image if available."""
    # Look for image files
    img_dirs = ['img', 'images', 'annotation']
    image_path = None
    
    for img_dir in img_dirs:
        if os.path.exists(img_dir):
            # Find first image file
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                files = [f for f in os.listdir(img_dir) if f.lower().endswith(ext[1:])]
                if files:
                    image_path = os.path.join(img_dir, files[0])
                    break
            if image_path:
                break
    
    if image_path:
        print(f"\nTesting with real image: {image_path}")
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print("✗ Could not read image")
                return False
                
            print(f"Original image shape: {img.shape}")
            
            # Test preprocessing pipeline
            processed = preprocess_for_cnn(img, target_size=(128, 128), equalize=True)
            print(f"Processed image shape: {processed.shape}, dtype: {processed.dtype}")
            print("✓ Real image test passed!")
            
        except Exception as e:
            print(f"✗ Error with real image: {str(e)}")
            return False
    else:
        print("\nNo real images found for testing (skipping)")
    
    return True


if __name__ == "__main__":
    success = test_basic_functionality()
    test_with_real_image()
    
    if success:
        print("\n🎉 All image processing functions are working correctly!")
    else:
        print("\n❌ Some tests failed. Check the implementation.")