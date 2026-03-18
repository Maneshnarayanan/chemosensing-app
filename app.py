from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemosense.color_science import get_color_parameters
from chemosense.image_io import validate_image
from chemosense.roi_extraction import extract_rgb_mean


st.set_page_config(page_title="ChemoSense - Mobile Ready", layout="centered")

MOBILE_CANVAS_WIDTH = 420


def init_state() -> None:
    if "blank_reference" not in st.session_state:
        st.session_state.blank_reference = None
    if "sample_results" not in st.session_state:
        st.session_state.sample_results = []
    if "sample_upload_nonce" not in st.session_state:
        st.session_state.sample_upload_nonce = 0


def load_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    if image is None:
        raise ValueError("Could not decode the selected image.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    validate_image(image_rgb)
    return image_rgb


def get_image_source(section_key: str, title: str):
    st.markdown(f"#### {title}")
    source = st.radio(
        "Image Source",
        options=["Camera", "Upload"],
        horizontal=True,
        key=f"{section_key}_source",
        label_visibility="collapsed",
    )

    if source == "Camera":
        return st.camera_input(
            "Take a photo",
            key=f"{section_key}_camera",
        )

    return st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        key=f"{section_key}_upload",
    )


def render_roi_selector(
    image_rgb: np.ndarray,
    roi_size: int,
    stroke_width: int,
    canvas_key: str,
    helper_text: str,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    height, width, _ = image_rgb.shape
    display_width = min(width, MOBILE_CANVAS_WIDTH)
    scale_factor = display_width / width
    display_height = max(int(height * scale_factor), 1)
    display_image = cv2.resize(image_rgb, (display_width, display_height))

    st.caption(helper_text)
    canvas_result = st_canvas(
        fill_color="rgba(46, 139, 87, 0.18)",
        stroke_width=stroke_width,
        stroke_color="#2E8B57",
        background_color="rgba(0,0,0,0)",
        background_image=Image.fromarray(display_image).convert("RGBA"),
        update_streamlit=True,
        height=display_image.shape[0],
        width=display_image.shape[1],
        drawing_mode="point",
        point_display_radius=max(int((roi_size * scale_factor) / 2), 8),
        key=canvas_key,
    )

    if canvas_result.json_data is None:
        return None, None

    objects = pd.json_normalize(canvas_result.json_data.get("objects", []))
    if objects.empty:
        return None, None

    last_point = objects.iloc[-1]
    cx = int(last_point["left"] / scale_factor)
    cy = int(last_point["top"] / scale_factor)
    roi_coords = (cx - roi_size // 2, cy - roi_size // 2, roi_size, roi_size)

    x, y, w, h = roi_coords
    if x < 0 or y < 0 or x + w > width or y + h > height:
        st.error("ROI is outside the image. Tap closer to the center of the sensing spot.")
        return None, None

    rgb = extract_rgb_mean(image_rgb, roi_coords)
    st.caption(
        f"ROI center: ({cx}, {cy}) | Mean RGB: ({rgb[0]:.1f}, {rgb[1]:.1f}, {rgb[2]:.1f})"
    )
    return rgb, roi_coords


def clear_samples() -> None:
    st.session_state.sample_results = []
    st.session_state.sample_upload_nonce += 1


def clear_all() -> None:
    st.session_state.blank_reference = None
    st.session_state.sample_results = []
    st.session_state.sample_upload_nonce += 1


init_state()

st.title("ChemoSense")
st.caption("Mobile-first blank and sample workflow")

with st.expander("Capture Guide", expanded=False):
    st.markdown(
        """
        - Keep the phone directly above the sensor.
        - Avoid glare, shadows, and flash reflections.
        - Use the same lighting for blank and samples.
        - Tap near the center of the sensing spot.
        """
    )

st.subheader("Settings")
roi_size = st.slider("ROI Box Size", 20, 100, 40)
stroke_width = st.slider("Marker Thickness", 1, 5, 2)

st.divider()
st.subheader("1. Save Blank Reference")

blank_file = get_image_source("blank", "Blank Image")
if blank_file is not None:
    try:
        blank_image_rgb = load_uploaded_image(blank_file)
        blank_rgb, blank_roi = render_roi_selector(
            blank_image_rgb,
            roi_size=roi_size,
            stroke_width=stroke_width,
            canvas_key="blank_canvas",
            helper_text="Tap once on the blank/reference spot.",
        )

        if blank_rgb is not None:
            blank_params = get_color_parameters(blank_rgb)
            blank_preview = pd.DataFrame(
                [
                    {
                        "RGB": f"({blank_rgb[0]:.1f}, {blank_rgb[1]:.1f}, {blank_rgb[2]:.1f})",
                        "L*": blank_params["L*"],
                        "a*": blank_params["a*"],
                        "b*": blank_params["b*"],
                    }
                ]
            )
            st.dataframe(
                blank_preview.style.format({"L*": "{:.2f}", "a*": "{:.2f}", "b*": "{:.2f}"}),
                use_container_width=True,
            )

            if st.button("Save Blank Reference", type="primary", use_container_width=True):
                st.session_state.blank_reference = {
                    "rgb": blank_rgb.tolist(),
                    "roi": list(blank_roi),
                }
                st.success("Blank saved. You can add samples now.")
    except Exception as exc:
        st.error(f"Blank image error: {exc}")

if st.session_state.blank_reference is not None:
    blank_rgb = st.session_state.blank_reference["rgb"]
    st.info(
        "Current blank: "
        f"RGB ({blank_rgb[0]:.1f}, {blank_rgb[1]:.1f}, {blank_rgb[2]:.1f})"
    )

st.divider()
st.subheader("2. Add Sample")

if st.session_state.blank_reference is None:
    st.warning("Save the blank reference first.")
else:
    sample_name = st.text_input(
        "Sample Name",
        value=f"Sample {len(st.session_state.sample_results) + 1}",
    )
    concentration_provided = st.toggle("Store known concentration", value=False)
    sample_concentration = st.number_input(
        "Known Concentration",
        min_value=0.0,
        value=0.0,
        step=0.1,
        disabled=not concentration_provided,
    )

    sample_file = get_image_source(
        f"sample_{st.session_state.sample_upload_nonce}",
        "Sample Image",
    )
    if sample_file is not None:
        try:
            sample_image_rgb = load_uploaded_image(sample_file)
            sample_rgb, sample_roi = render_roi_selector(
                sample_image_rgb,
                roi_size=roi_size,
                stroke_width=stroke_width,
                canvas_key=f"sample_canvas_{st.session_state.sample_upload_nonce}",
                helper_text="Tap once on the sample spot.",
            )

            if sample_rgb is not None:
                blank_rgb = np.array(st.session_state.blank_reference["rgb"], dtype=float)
                params = get_color_parameters(sample_rgb, reference_rgb=blank_rgb)

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("L*", f"{params['L*']:.2f}")
                metric_col2.metric("a*", f"{params['a*']:.2f}")
                metric_col3.metric("Delta E", f"{params['delta_e']:.2f}")

                with st.expander("Sample Details", expanded=False):
                    preview_df = pd.DataFrame(
                        [
                            {
                                "Sample": sample_name,
                                "RGB": f"({sample_rgb[0]:.1f}, {sample_rgb[1]:.1f}, {sample_rgb[2]:.1f})",
                                "L*": params["L*"],
                                "a*": params["a*"],
                                "b*": params["b*"],
                                "dL*": params["dL*"],
                                "da*": params["da*"],
                                "db*": params["db*"],
                                "delta_e": params["delta_e"],
                            }
                        ]
                    )
                    st.dataframe(
                        preview_df.style.format(
                            {
                                "L*": "{:.2f}",
                                "a*": "{:.2f}",
                                "b*": "{:.2f}",
                                "dL*": "{:.2f}",
                                "da*": "{:.2f}",
                                "db*": "{:.2f}",
                                "delta_e": "{:.2f}",
                            }
                        ),
                        use_container_width=True,
                    )

                if st.button("Add Sample", type="primary", use_container_width=True):
                    st.session_state.sample_results.append(
                        {
                            "sample_name": sample_name,
                            "image_name": sample_file.name,
                            "roi_x": int(sample_roi[0]),
                            "roi_y": int(sample_roi[1]),
                            "roi_w": int(sample_roi[2]),
                            "roi_h": int(sample_roi[3]),
                            "R": float(params["R"]),
                            "G": float(params["G"]),
                            "B": float(params["B"]),
                            "L*": float(params["L*"]),
                            "a*": float(params["a*"]),
                            "b*": float(params["b*"]),
                            "dL*": float(params["dL*"]),
                            "da*": float(params["da*"]),
                            "db*": float(params["db*"]),
                            "delta_e": float(params["delta_e"]),
                            "concentration": float(sample_concentration) if concentration_provided else np.nan,
                        }
                    )
                    st.session_state.sample_upload_nonce += 1
                    st.success(f"{sample_name} added.")
                    st.rerun()
        except Exception as exc:
            st.error(f"Sample image error: {exc}")

st.divider()
st.subheader("3. Session Results")

results = st.session_state.sample_results
if not results:
    st.info("No samples added yet.")
else:
    results_df = pd.DataFrame(results)

    summary_col1, summary_col2 = st.columns(2)
    summary_col1.metric("Samples", len(results_df))
    summary_col2.metric("Mean Delta E", f"{results_df['delta_e'].mean():.2f}")

    for _, row in results_df.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['sample_name']}**")
            st.caption(row["image_name"])
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            stat_col1.metric("Delta E", f"{row['delta_e']:.2f}")
            stat_col2.metric("L*", f"{row['L*']:.2f}")
            stat_col3.metric("a*", f"{row['a*']:.2f}")
            if not pd.isna(row["concentration"]):
                st.caption(f"Concentration: {row['concentration']:.2f}")

    with st.expander("Detailed Table", expanded=False):
        display_df = results_df.copy()
        display_df["RGB"] = display_df.apply(
            lambda row: f"({row['R']:.1f}, {row['G']:.1f}, {row['B']:.1f})",
            axis=1,
        )
        ordered_cols = [
            "sample_name",
            "image_name",
            "RGB",
            "L*",
            "a*",
            "b*",
            "dL*",
            "da*",
            "db*",
            "delta_e",
            "concentration",
        ]
        st.dataframe(
            display_df[ordered_cols].style.format(
                {
                    "L*": "{:.2f}",
                    "a*": "{:.2f}",
                    "b*": "{:.2f}",
                    "dL*": "{:.2f}",
                    "da*": "{:.2f}",
                    "db*": "{:.2f}",
                    "delta_e": "{:.2f}",
                    "concentration": "{:.2f}",
                },
                na_rep="-",
            ),
            use_container_width=True,
        )

    fig_bar, ax_bar = plt.subplots(figsize=(6, 3.5))
    ax_bar.bar(results_df["sample_name"], results_df["delta_e"], color="#2E8B57")
    ax_bar.set_ylabel("Delta E")
    ax_bar.set_xlabel("Samples")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)

    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="chemosense_session_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

action_col1, action_col2 = st.columns(2)
with action_col1:
    if st.button("Clear Samples", use_container_width=True):
        clear_samples()
        st.rerun()

with action_col2:
    if st.button("Reset All", use_container_width=True):
        clear_all()
        st.rerun()
