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

from chemosense.reporting import (
    generate_pdf_report,
    generate_html_report,
    create_annotated_thumbnail,
    create_spot_crop,
)
import matplotlib.pyplot as plt

def test_reporting():
    print("\nTesting Reporting Generation & Image Details...")
    dummy_img = np.ones((200, 200, 3), dtype=np.uint8) * 180
    roi = (20, 20, 40, 40)
    annotated_thumb = create_annotated_thumbnail(dummy_img, roi)
    spot_thumb = create_spot_crop(dummy_img, roi)
    assert len(annotated_thumb) > 100
    assert len(spot_thumb) > 100

    df = pd.DataFrame([
        {
            "sample_name": "Sample 1",
            "image_name": "test.png",
            "image_shape": [200, 200, 3],
            "roi_x": 20,
            "roi_y": 20,
            "roi_w": 40,
            "roi_h": 40,
            "R": 150.0,
            "G": 100.0,
            "B": 100.0,
            "L*": 47.6,
            "a*": 20.2,
            "b*": 8.3,
            "dL*": -33.0,
            "da*": 20.2,
            "db*": 8.3,
            "delta_e": 39.5,
            "concentration": 10.0,
            "annotated_thumb": annotated_thumb,
            "spot_thumb": spot_thumb,
        }
    ])
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.bar(df["sample_name"], df["delta_e"])
    blank_ref = {
        "image_name": "blank.png",
        "image_shape": [200, 200, 3],
        "rgb": [200.0, 200.0, 200.0],
        "roi": [10, 10, 40, 40],
        "annotated_thumb": annotated_thumb,
        "spot_thumb": spot_thumb,
    }

    pdf_bytes = generate_pdf_report(df, blank_ref, fig)
    assert len(pdf_bytes) > 2000
    print(f"PDF Report generated with Image Details: {len(pdf_bytes)} bytes")

    html_str = generate_html_report(df, blank_ref, fig)
    assert "<html" in html_str and "Sample 1" in html_str and "data:image/jpeg;base64" in html_str
    print(f"HTML Report generated with Image Details: {len(html_str)} chars")
    print("Reporting Test Passed!")

if __name__ == "__main__":
    try:
        test_color_conversion()
        test_quantification()
        test_reporting()
        print("\nAll Core Tests Passed Successfully!")
    except Exception as e:
        print(f"\nTest Failed: {e}")
        sys.exit(1)

