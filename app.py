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

from streamlit_drawable_canvas import st_canvas
import io
import base64
from PIL import Image

st.set_page_config(page_title="ChemoSense - Image-Based Sensing", layout="wide")

st.title("🧪 ChemoSense")
st.markdown("### Computer-Assisted Image-Based Chemosensing System")

with st.sidebar:
    st.header("Settings")
    roi_size = st.slider("ROI Box Size", 10, 100, 40)
    stroke_width = st.slider("Stroke width", 1, 5, 2)
    
    st.divider()
    st.markdown("#### Instructions")
    st.info(
        """
        1. Upload an image.
        2. Draw **points** (click) on the centers of your spots.
        3. The **first point** is considered the **Blank**.
        4. Subsequent points are **Samples**.
        5. Assign concentrations in the generated table.
        """
    )
    
uploaded_file = st.file_uploader("Upload Sensor Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize for display if too large (improves performance and visibility)
    max_display_width = 800
    h, w, c = image_rgb.shape
    if w > max_display_width:
        scale_factor = max_display_width / w
        new_h = int(h * scale_factor)
        display_image = cv2.resize(image_rgb, (max_display_width, new_h))
    else:
        scale_factor = 1.0
        display_image = image_rgb
    
    col_img, col_res = st.columns([1.5, 1])
    
    with col_img:
        st.subheader("Region Selection")
        
        # Simple PIL Image pass - st_canvas handles resizing calculation
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color="#00FF00",
            background_color="rgba(0,0,0,0)",
            background_image=Image.fromarray(display_image).convert("RGBA"),
            update_streamlit=True,
            height=display_image.shape[0],
            width=display_image.shape[1],
            drawing_mode="point",
            point_display_radius=roi_size // 2, # Visual radius on canvas
            key=f"canvas_{uploaded_file.name}_{display_image.shape}", # Reset canvas if file changes
        )
        
    with col_res:
        st.subheader("Analysis Results")
        
        if canvas_result.json_data is not None:
            objects = pd.json_normalize(canvas_result.json_data["objects"]) 
            
            if not objects.empty:
                # Collect data first
                temp_results = []
                
                # Iterate through drawn points
                for i, row in objects.iterrows():
                    # Center coordinates from canvas (need to scale back to original)
                    cx_disp, cy_disp = row["left"], row["top"]
                    
                    # Scale back to original image coordinates
                    cx = int(cx_disp / scale_factor)
                    cy = int(cy_disp / scale_factor)
                    
                    # Define ROI on original image
                    roi_coords = (cx - roi_size//2, cy - roi_size//2, roi_size, roi_size)
                    
                    try:
                        # Extract from original high-res image
                        rgb = extract_rgb_mean(image_rgb, roi_coords)
                        temp_results.append({
                            'id': i,
                            'Name': "Blank" if i == 0 else f"Sample {i}",
                            'Concentration': 0.0 if i == 0 else float(i)*10,
                            'RGB': rgb
                        })
                    except Exception as e:
                        st.warning(f"Skipped point at ({cx},{cy}): ROI out of bounds?")
                
                if temp_results:
                    # Create DataFrame for editing
                    input_df = pd.DataFrame([{'Name': r['Name'], 'Concentration': r['Concentration']} for r in temp_results])
                    
                    st.markdown("### Edit Sample Details")
                    edited_df = st.data_editor(input_df, use_container_width=True)
                    
                    # Process results with edited values
                    final_results = []
                    blank_rgb = temp_results[0]['RGB'] # Assume first point is always blank for calculation reference
                    
                    for idx, row in edited_df.iterrows():
                        # Match back to RGB data (assuming order hasn't changed, which it shouldn't in data_editor unless rows deleted)
                        # To be safe, we could use the index if user doesn't delete rows. 
                        # For simple impl, we assume 1:1 mapping by index.
                        if idx < len(temp_results):
                            rgb = temp_results[idx]['RGB']
                            params = get_color_parameters(rgb, reference_rgb=blank_rgb)
                            params['name'] = row['Name']
                            params['conc'] = row['Concentration']
                            final_results.append(params)
                    
                    df = pd.DataFrame(final_results)
                    
                    # CSV Export
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name='chemosense_results.csv',
                        mime='text/csv',
                    )

                    # Visualization
                    if len(df) > 1:
                        st.markdown("#### Analysis Plots")
                        
                        tab1, tab2 = st.tabs(["Bar Chart (Selectivity)", "Calibration Curve"])
                        
                        with tab1:
                            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                            # Filter out blank if needed, or simply plot all. 
                            # Usually Selectivity plots show samples. Let's plot all except maybe Blank if it's 0 Delta E.
                            plot_df = df.iloc[1:] if len(df) > 1 else df
                            
                            bars = ax_bar.bar(plot_df['name'], plot_df['delta_e'], color=plt.cm.hsv(np.linspace(0, 1, len(plot_df))))
                            
                            ax_bar.set_ylabel("ΔE", fontsize=12, fontweight='bold')
                            ax_bar.set_xlabel("Metal ions", fontsize=12, fontweight='bold')
                            
                            # Customize ticks
                            plt.xticks(fontsize=10, fontweight='bold')
                            plt.yticks(fontsize=10)
                            
                            # Remove top and right spines for cleaner look similar to image
                            ax_bar.spines['right'].set_visible(True) # Image has box
                            ax_bar.spines['top'].set_visible(True)
                            
                            st.pyplot(fig_bar)

                        with tab2:
                            # Calibration (need at least 2 points including blank)
                            model_results = train_calibration_model(df['delta_e'].values, df['conc'].values)
                            
                            fig, ax = plt.subplots(figsize=(4, 3))
                            ax.scatter(df['delta_e'], df['conc'], color='blue', label='Points')
                            
                            x_range = np.linspace(df['delta_e'].min(), df['delta_e'].max(), 50)
                            y_pred = predict_concentration(model_results, x_range)
                            ax.plot(x_range, y_pred, color='red', linestyle='--', label=f"R²={model_results['metrics']['R2']:.3f}")
                            
                            ax.set_xlabel("Delta E")
                            ax.set_ylabel("Concentration")
                            ax.legend()
                            st.pyplot(fig)
            else:
                st.info("Click on the image to select points. The first click is the Blank.")

else:
    st.info("Please upload a sensor image to start analysis.")

