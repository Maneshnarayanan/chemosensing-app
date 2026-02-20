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
    
    with st.expander("Formula reference"):
        st.latex(r"\Delta L^* = L^*_{sample} - L^*_{blank}")
        st.latex(r"\Delta a^* = a^*_{sample} - a^*_{blank}")
        st.latex(r"\Delta b^* = b^*_{sample} - b^*_{blank}")
        st.latex(r"\Delta E = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}")
    
uploaded_file = st.file_uploader("Upload Sensor Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize for display if too large (improves performance and visibility)
    # Display settings
    max_display_width = 1000
    h, w, c = image_rgb.shape
    if w > max_display_width:
        scale_factor = max_display_width / w
        new_h = int(h * scale_factor)
        display_image = cv2.resize(image_rgb, (max_display_width, new_h))
    else:
        scale_factor = 1.0
        display_image = image_rgb
    
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
    
    st.divider()
    
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
                        'RGB': rgb
                    })
                except Exception as e:
                    st.warning(f"Skipped point at ({cx},{cy}): ROI out of bounds?")
                
            if temp_results:
                # Create DataFrame for editing
                input_df = pd.DataFrame([{'Name': r['Name']} for r in temp_results])
                
                st.markdown("### Edit Sample Names")
                edited_df = st.data_editor(input_df, use_container_width=True)
                
                # Process results with edited values
                final_results = []
                blank_rgb = temp_results[0]['RGB'] # Assume first point is always blank for calculation reference
                
                for idx, row in edited_df.iterrows():
                    # Match back to RGB data (assuming order hasn't changed, which it shouldn't in data_editor unless rows deleted)
                    # To be safe, we could use the index if user doesn't delete rows. 
                    # For simple impl, we assume 1:1 mapping by index.
                    if int(idx) < len(temp_results):
                        rgb = temp_results[idx]['RGB']
                        params = get_color_parameters(rgb, reference_rgb=blank_rgb)
                        params['name'] = row['Name']
                        # Add RGB values formatted as string
                        params['RGB'] = f"({rgb[0]:.1f}, {rgb[1]:.1f}, {rgb[2]:.1f})"
                        final_results.append(params)
                
                df = pd.DataFrame(final_results)
                
                # Reorder columns for better readability
                cols = ['name', 'RGB', 'L*', 'a*', 'b*', 'dL*', 'da*', 'db*', 'delta_e']
                # Keep only existing columns to avoid key errors if some aren't present
                df = df[[c for c in cols if c in df.columns]]
                
                st.markdown("### Analysis Summary Table")
                st.dataframe(df.style.format({
                    'L*': "{:.2f}", 'a*': "{:.2f}", 'b*': "{:.2f}",
                    'dL*': "{:.2f}", 'da*': "{:.2f}", 'db*': "{:.2f}",
                    'delta_e': "{:.2f}"
                }), use_container_width=True)
                
                # CSV Export
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Results as CSV",
                    data=csv,
                    file_name='chemosense_results.csv',
                    mime='text/csv',
                )

                # Visualization
                if len(df) > 1:
                    st.markdown("#### Analysis Plots")
                    
                    # Bar Chart (Selectivity)
                    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
                    plot_df = df.iloc[1:] if len(df) > 1 else df
                    
                    ax_bar.bar(plot_df['name'], plot_df['delta_e'], color=plt.cm.hsv(np.linspace(0, 1, len(plot_df))))
                    
                    ax_bar.set_ylabel("ΔE", fontsize=12, fontweight='bold')
                    ax_bar.set_xlabel("Metal ions", fontsize=12, fontweight='bold')
                    
                    plt.xticks(fontsize=10, fontweight='bold')
                    plt.yticks(fontsize=10)
                    
                    st.pyplot(fig_bar)
        else:
            st.info("Click on the image to select points. The first click is the Blank.")

else:
    st.info("Please upload a sensor image to start analysis.")
