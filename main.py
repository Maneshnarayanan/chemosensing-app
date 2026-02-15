import os
import sys
import numpy as np
import cv2
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath("src"))

from chemosense.image_io import load_image
from chemosense.roi_extraction import extract_rgb_mean
from chemosense.color_science import get_color_parameters
from chemosense.quantification import train_calibration_model, predict_concentration

def generate_synthetic_sensor_image(output_path: str):
    """Generates a synthetic sensor image with 4 spots of different colors."""
    # Create a 400x400 white background
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Centers for 4 spots
    spots = [
        {"name": "Blank", "center": (100, 100), "color": (200, 200, 200)}, # Light gray
        {"name": "Sample1", "center": (300, 100), "color": (150, 100, 100)}, # Reddish
        {"name": "Sample2", "center": (100, 300), "color": (100, 150, 100)}, # Greenish
        {"name": "Sample3", "center": (300, 300), "color": (100, 100, 150)}, # Bluish
    ]
    
    for spot in spots:
        # Draw a filled circle
        cv2.circle(img, spot["center"], 40, spot["color"][::-1], -1) # BGR for OpenCV
        
    cv2.imwrite(output_path, img)
    return spots

def run_app():
    print("=== Chemosensing System Demonstration ===")
    
    # 1. Setup Data
    image_path = "synthetic_sensor.png"
    print(f"Generating synthetic sensor image: {image_path}")
    spots_def = generate_synthetic_sensor_image(image_path)
    
    # 2. Load Image
    print("Loading image...")
    image = load_image(image_path)
    
    # 3. Extract RGB and Convert to Color Parameters
    print("Extracting color data from ROIs...")
    results = []
    # Define ROIs (x, y, w, h) based on spot centers
    rois = {
        "Blank": (100-20, 100-20, 40, 40),
        "Sample1": (300-20, 100-20, 40, 40),
        "Sample2": (100-20, 300-20, 40, 40),
        "Sample3": (300-20, 300-20, 40, 40),
    }
    
    # Reference RGB from Blank
    blank_rgb = extract_rgb_mean(image, rois["Blank"])
    
    for name, roi in rois.items():
        rgb = extract_rgb_mean(image, roi)
        params = get_color_parameters(rgb, reference_rgb=blank_rgb)
        params['spot'] = name
        results.append(params)
        
    df = pd.DataFrame(results)
    
    print("\nExtraction Results:")
    print(df[['spot', 'R', 'G', 'B', 'L*', 'a*', 'b*', 'delta_e']])
    
    # 4. Calibration & Prediction (Simplified)
    # Assume Sample 1, 2, 3 have known concentrations for this demo
    known_conc = np.array([0, 10, 20, 30]) # Blank=0
    df['conc'] = known_conc
    
    print("\nTraining calibration model (Delta E vs Concentration)...")
    model_data = train_calibration_model(df['delta_e'].values, df['conc'].values)
    print(f"Model R2: {model_data['metrics']['R2']:.4f}")
    
    # 5. Predict Unknown (Simulating a new spot)
    print("\nSimulating unknown sample prediction...")
    unknown_rgb = np.array([120, 120, 180]) # Slightly more blue than Sample 3
    unknown_params = get_color_parameters(unknown_rgb, reference_rgb=blank_rgb)
    predicted_conc = predict_concentration(model_data, np.array([unknown_params['delta_e']]))
    
    print(f"Unknown Signal (Delta E): {unknown_params['delta_e']:.2f}")
    print(f"Predicted Concentration: {predicted_conc[0]:.2f}")
    
    print("\nApplication run complete.")

if __name__ == "__main__":
    run_app()
