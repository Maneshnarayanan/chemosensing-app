import os
import cv2
import glob
import numpy as np
from typing import List, Union, Tuple

def load_image(file_path: str) -> np.ndarray:
    """Loads an image and validates its existence and depth."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Could not decode image: {file_path}")
    
    # Convert BGR (OpenCV default) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def validate_image(img: np.ndarray, min_resolution: Tuple[int, int] = (100, 100)) -> bool:
    """Validates image resolution and color depth."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must be a color image (3 channels).")
    
    if img.shape[0] < min_resolution[0] or img.shape[1] < min_resolution[1]:
        raise ValueError(f"Image resolution too low. Minimum required: {min_resolution}")
    
    return True

def batch_load_images(directory: str, extensions: List[str] = ["*.jpg", "*.png", "*.jpeg"]) -> List[Tuple[str, np.ndarray]]:
    """Loads all images from a directory matching the extensions."""
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    
    images = []
    for f in sorted(files):
        try:
            img = load_image(f)
            validate_image(img)
            images.append((os.path.basename(f), img))
        except Exception as e:
            print(f"Warning: Failed to load {f}. Error: {e}")
            
    return images
