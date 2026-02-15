import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath("src"))

from chemosense.color_science import srgb_to_lab, compute_delta_e_cie76
from chemosense.quantification import train_calibration_model, predict_concentration

def test_color_conversion():
    print("Testing Color Conversion...")
    # Pure White
    rgb_white = np.array([255, 255, 255])
    lab_white = srgb_to_lab(rgb_white)
    print(f"White LAB: {lab_white}")
    # Expected: L=100, a=0, b=0 (approx)
    assert np.allclose(lab_white, [100, 0, 0], atol=0.1)
    
    # Pure Red
    rgb_red = np.array([255, 0, 0])
    lab_red = srgb_to_lab(rgb_red)
    print(f"Red LAB: {lab_red}")
    
    # Delta E between white and red
    de = compute_delta_e_cie76(lab_white, lab_red)
    print(f"Delta E (White-Red): {de}")
    assert de > 0
    print("Color Conversion Test Passed!")

def test_quantification():
    print("\nTesting Quantification...")
    # Synthetic data: Signal = 2 * Conc + 1
    conc = np.array([0, 1, 2, 3, 4, 5])
    signals = 2 * conc + 1 + np.random.normal(0, 0.1, len(conc))
    
    model_results = train_calibration_model(signals, conc)
    print(f"Metrics: {model_results['metrics']}")
    assert model_results['metrics']['R2'] > 0.95
    
    test_signal = np.array([3.0]) # Should be (3-1)/2 = 1.0
    pred = predict_concentration(model_results, test_signal)
    print(f"Predicted for signal 3.0: {pred[0]}")
    assert np.allclose(pred, [1.0], atol=0.2)
    print("Quantification Test Passed!")

if __name__ == "__main__":
    try:
        test_color_conversion()
        test_quantification()
        print("\nAll Core Tests Passed Successfully!")
    except Exception as e:
        print(f"\nTest Failed: {e}")
        sys.exit(1)
