import streamlit as st
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image

# Add src to path
sys.path.append(os.path.abspath("src"))

from chemosense.image_io import validate_image
from chemosense.roi_extraction import extract_rgb_mean
from chemosense.color_science import get_color_parameters
from chemosense.quantification import train_calibration_model, predict_concentration

st.set_page_config(page_title="ChemoSense - Image-Based Sensing", layout="wide")

st.title("🧪 ChemoSense")
st.markdown("### Computer-Assisted Image-Based Chemosensing System")

with st.sidebar:
    st.header("Settings")
    roi_size = st.slider("ROI Box Size", 10, 100, 40)
    
    st.divider()
    st.markdown("#### Calibration Data")
    blank_x = st.number_input("Blank ROI X", value=100)
    blank_y = st.number_input("Blank ROI Y", value=100)
    
    st.divider()
    st.markdown("#### Sample ROIs")
    sample_count = st.number_input("Number of Samples", 1, 10, 3)
    
    sample_rois = []
    for i in range(sample_count):
        st.markdown(f"**Sample {i+1}**")
        col1, col2 = st.columns(2)
        with col1:
            x = st.number_input(f"X {i+1}", value=300 if i % 2 else 100, key=f"x_{i}")
        with col2:
            y = st.number_input(f"Y {i+1}", value=100 if i < 2 else 300, key=f"y_{i}")
        conc = st.number_input(f"Conc {i+1}", value=(i+1)*10.0, key=f"c_{i}")
        sample_rois.append({"x": x, "y": y, "conc": conc, "name": f"Sample {i+1}"})

uploaded_file = st.file_uploader("Upload Sensor Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    col_img, col_res = st.columns([1, 1])
    
    with col_img:
        st.subheader("Image Analysis")
        # Draw ROIs on a copy
        preview = image_rgb.copy()
        
        # Blank ROI
        cv2.rectangle(preview, (blank_x-roi_size//2, blank_y-roi_size//2), 
                      (blank_x+roi_size//2, blank_y+roi_size//2), (255, 255, 255), 2)
        cv2.putText(preview, "Blank", (blank_x-20, blank_y-roi_size//2-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Sample ROIs
        for i, s in enumerate(sample_rois):
            cv2.rectangle(preview, (s['x']-roi_size//2, s['y']-roi_size//2), 
                          (s['x']+roi_size//2, s['y']+roi_size//2), (0, 255, 0), 2)
            cv2.putText(preview, f"S{i+1}", (s['x']-10, s['y']-roi_size//2-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        st.image(preview, use_container_width=True, caption="Sensor Image with ROIs")
        
    with col_res:
        st.subheader("Results")
        
        # Process data
        results = []
        # Reference (Blank)
        blank_roi_coords = (blank_x-roi_size//2, blank_y-roi_size//2, roi_size, roi_size)
        blank_rgb = extract_rgb_mean(image_rgb, blank_roi_coords)
        
        # Add Blank to results
        blank_params = get_color_parameters(blank_rgb, reference_rgb=blank_rgb)
        blank_params['name'] = "Blank"
        blank_params['conc'] = 0.0
        results.append(blank_params)
        
        # Samples
        for s in sample_rois:
            roi_coords = (s['x']-roi_size//2, s['y']-roi_size//2, roi_size, roi_size)
            rgb = extract_rgb_mean(image_rgb, roi_coords)
            params = get_color_parameters(rgb, reference_rgb=blank_rgb)
            params['name'] = s['name']
            params['conc'] = s['conc']
            results.append(params)
            
        df = pd.DataFrame(results)
        st.dataframe(df[['name', 'R', 'G', 'B', 'L*', 'a*', 'b*', 'delta_e', 'conc']], use_container_width=True)
        
        # Calibration
        st.markdown("#### Calibration Curve")
        cal_data = df.iloc[1:] # Exclude Blank if desired, or include 0
        model_results = train_calibration_model(df['delta_e'].values, df['conc'].values)
        
        fig, ax = plt.subplots()
        ax.scatter(df['delta_e'], df['conc'], color='blue', label='Standard Points')
        
        # Trend line
        x_range = np.linspace(df['delta_e'].min(), df['delta_e'].max(), 50)
        y_pred = predict_concentration(model_results, x_range)
        ax.plot(x_range, y_pred, color='red', linestyle='--', label=f"Fit (R²={model_results['metrics']['R2']:.3f})")
        
        ax.set_xlabel("Delta E (Color Shift)")
        ax.set_ylabel("Concentration")
        ax.legend()
        st.pyplot(fig)

    # Unknown Prediction
    st.divider()
    st.subheader("Predict Unknown Sample")
    u_col1, u_col2 = st.columns(2)
    with u_col1:
        u_x = st.number_input("Unknown ROI X", value=200)
    with u_col2:
        u_y = st.number_input("Unknown ROI Y", value=200)
    
    if st.button("Predict Concentration"):
        u_roi_coords = (u_x-roi_size//2, u_y-roi_size//2, roi_size, roi_size)
        u_rgb = extract_rgb_mean(image_rgb, u_roi_coords)
        u_params = get_color_parameters(u_rgb, reference_rgb=blank_rgb)
        
        pred = predict_concentration(model_results, np.array([u_params['delta_e']]))
        
        st.success(f"**Predicted Concentration: {pred[0]:.2f}** (Signal ΔE: {u_params['delta_e']:.2f})")
        
else:
    st.info("Please upload a sensor image to start analysis. Use the sidebar to adjust ROI coordinates.")
    
    # Show example button
    if st.button("Use Sample Image"):
        st.info("Use the 'synthetic_sensor.png' generated previously for a quick test.")
