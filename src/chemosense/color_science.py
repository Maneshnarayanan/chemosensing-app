import numpy as np
from skimage import color
from typing import Union, Dict

def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Converts sRGB values (0-255) to CIELAB using D65 illuminant.
    Input can be a single RGB vector or an array of RGB vectors.
    """
    # Normalize to 0-1
    rgb_norm = rgb / 255.0
    
    # If single vector, reshape for skimage
    is_single = rgb_norm.ndim == 1
    if is_single:
        rgb_norm = rgb_norm.reshape(1, 1, 3)
    else:
        # Assume (N, 3)
        rgb_norm = rgb_norm.reshape(-1, 1, 3)
        
    lab = color.rgb2lab(rgb_norm)
    
    if is_single:
        return lab.flatten()
    else:
        return lab.reshape(-1, 3)

def compute_delta_e_cie76(lab_sample: np.ndarray, lab_reference: np.ndarray) -> Union[float, np.ndarray]:
    """
    Computes ΔE using the CIE76 formula (Euclidean distance in CIELAB).
    """
    diff = lab_sample - lab_reference
    delta_e = np.sqrt(np.sum(diff**2, axis=-1))
    return delta_e

def get_color_parameters(rgb: np.ndarray, reference_rgb: np.ndarray = None) -> Dict[str, Union[float, np.ndarray]]:
    """
    Returns a dictionary of color parameters for a given RGB and optional reference.
    """
    lab = srgb_to_lab(rgb)
    results = {
        'R': rgb[0], 'G': rgb[1], 'B': rgb[2],
        'L*': lab[0], 'a*': lab[1], 'b*': lab[2]
    }
    
    if reference_rgb is not None:
        lab_ref = srgb_to_lab(reference_rgb)
        results['delta_e'] = compute_delta_e_cie76(lab, lab_ref)
        
    return results
