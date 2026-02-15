import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

def extract_rgb_mean(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Extracts mean RGB values from an image based on ROI (x, y, w, h).
    """
    x, y, w, h = roi
    roi_img = image[y:y+h, x:x+w]
    
    if roi_img.size == 0:
        raise ValueError("Selected ROI is empty. Check coordinates.")
        
    mean_rgb = np.mean(roi_img, axis=(0, 1))
    return mean_rgb

def extract_batch_rgb(images: List[Tuple[str, np.ndarray]], roi: Tuple[int, int, int, int]) -> pd.DataFrame:
    """
    Extracts mean RGB values for a batch of images and returns a DataFrame.
    """
    data = []
    for filename, img in images:
        mean_rgb = extract_rgb_mean(img, roi)
        data.append({
            'filename': filename,
            'R': mean_rgb[0],
            'G': mean_rgb[1],
            'B': mean_rgb[2]
        })
        
    return pd.DataFrame(data)

def extract_multi_roi(image: np.ndarray, rois: Dict[str, Tuple[int, int, int, int]]) -> Dict[str, np.ndarray]:
    """
    Extracts mean RGB values for multiple ROIs (named spots) in a single image.
    """
    results = {}
    for name, roi in rois.items():
        results[name] = extract_rgb_mean(image, roi)
    return results
