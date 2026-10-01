import io
import base64
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch


def fig_to_bytes(fig: plt.Figure) -> bytes:
    """Converts a matplotlib Figure into PNG byte buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def create_annotated_thumbnail(image_rgb: np.ndarray, roi: Tuple[int, int, int, int], max_dim: int = 360) -> str:
    """Draws ROI on image, resizes maintaining aspect ratio, and returns base64 JPEG."""
    h, w, _ = image_rgb.shape
    annotated = image_rgb.copy()
    rx, ry, rw, rh = roi

    thickness = max(2, int(min(h, w) / 160))
    cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (46, 204, 113), thickness)

    scale = max_dim / max(h, w)
    if scale < 1.0:
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)
        annotated = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buf).decode("utf-8")


def create_spot_crop(image_rgb: np.ndarray, roi: Tuple[int, int, int, int], margin_ratio: float = 0.35, out_size: int = 120) -> str:
    """Crops the sensing spot with a margin, resizes, and returns base64 JPEG."""
    h, w, _ = image_rgb.shape
    rx, ry, rw, rh = roi

    mx = int(rw * margin_ratio)
    my = int(rh * margin_ratio)

    x1 = max(0, rx - mx)
    y1 = max(0, ry - my)
    x2 = min(w, rx + rw + mx)
    y2 = min(h, ry + rh + my)

    crop = image_rgb[y1:y2, x1:x2].copy()
    if crop.size == 0:
        crop = image_rgb[ry : ry + rh, rx : rx + rw].copy()

    crop_resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    crop_bgr = cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf).decode("utf-8")


def generate_pdf_report(
    results_df: pd.DataFrame,
    blank_ref: Optional[Dict[str, Any]],
    chart_fig: Optional[plt.Figure] = None,
    report_title: str = "ChemoSense Analytical Report",
) -> bytes:
    """Generates a comprehensive PDF report containing metadata, blank ref, data table, graph, and captured image gallery."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E3A8A"),
        alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#64748B"),
    )
    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1E293B"),
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0F172A"),
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
    )

    story = []

    # Header Title
    story.append(Paragraph(report_title, title_style))
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(
        Paragraph(
            f"Generated on {now_str} &bull; Transduction Method: CIELAB ΔE (CIE76 / D65)",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=10))

    # Section 1: Overview & Blank Reference
    story.append(Paragraph("1. Session Overview & Calibration Baseline", h2_style))

    blank_rgb_str = "N/A"
    blank_roi_str = "N/A"
    blank_file_str = "N/A"
    if blank_ref:
        b_rgb = blank_ref.get("rgb", [0, 0, 0])
        b_roi = blank_ref.get("roi", [0, 0, 0, 0])
        blank_rgb_str = f"({b_rgb[0]:.1f}, {b_rgb[1]:.1f}, {b_rgb[2]:.1f})"
        blank_roi_str = f"x={b_roi[0]}, y={b_roi[1]}, w={b_roi[2]}, h={b_roi[3]}"
        blank_file_str = str(blank_ref.get("image_name", "blank_image"))

    mean_de = f"{results_df['delta_e'].mean():.2f}" if not results_df.empty else "N/A"
    max_de = f"{results_df['delta_e'].max():.2f}" if not results_df.empty else "N/A"
    min_de = f"{results_df['delta_e'].min():.2f}" if not results_df.empty else "N/A"

    summary_data = [
        [
            Paragraph("<b>Blank Reference RGB:</b>", body_style),
            Paragraph(blank_rgb_str, body_style),
            Paragraph("<b>Total Samples:</b>", body_style),
            Paragraph(str(len(results_df)), body_style),
        ],
        [
            Paragraph("<b>Blank Source File:</b>", body_style),
            Paragraph(blank_file_str, body_style),
            Paragraph("<b>Mean ΔE:</b>", body_style),
            Paragraph(mean_de, body_style),
        ],
        [
            Paragraph("<b>Blank ROI Box:</b>", body_style),
            Paragraph(blank_roi_str, body_style),
            Paragraph("<b>ΔE Range (Min - Max):</b>", body_style),
            Paragraph(f"{min_de} &ndash; {max_de}", body_style),
        ],
    ]

    summary_table = Table(summary_data, colWidths=[130, 140, 110, 160])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # Section 2: Visualizations (Chart)
    if chart_fig is not None:
        story.append(Paragraph("2. Quantitative Colorimetric Response (ΔE Chart)", h2_style))
        img_data = fig_to_bytes(chart_fig)
        img_buf = io.BytesIO(img_data)
        chart_img = RLImage(img_buf, width=6.8 * inch, height=2.9 * inch)
        story.append(chart_img)
        story.append(Spacer(1, 10))

    # Section 3: Detailed Data Table
    story.append(Paragraph("3. Analytical Measurements Data Table", h2_style))

    if results_df.empty:
        story.append(Paragraph("No sample data recorded in this session.", body_style))
    else:
        headers = ["Sample", "Image", "Mean RGB", "L*", "a*", "b*", "ΔL*", "Δa*", "Δb*", "ΔE", "Conc."]
        table_rows = [[Paragraph(h, table_header_style) for h in headers]]

        for _, row in results_df.iterrows():
            rgb_text = f"({row['R']:.0f},{row['G']:.0f},{row['B']:.0f})"
            conc_val = f"{row['concentration']:.2f}" if not pd.isna(row.get("concentration")) else "-"
            table_rows.append(
                [
                    Paragraph(str(row.get("sample_name", "")), table_cell_style),
                    Paragraph(str(row.get("image_name", "")), table_cell_style),
                    Paragraph(rgb_text, table_cell_style),
                    Paragraph(f"{row.get('L*', 0):.1f}", table_cell_style),
                    Paragraph(f"{row.get('a*', 0):.1f}", table_cell_style),
                    Paragraph(f"{row.get('b*', 0):.1f}", table_cell_style),
                    Paragraph(f"{row.get('dL*', 0):.1f}", table_cell_style),
                    Paragraph(f"{row.get('da*', 0):.1f}", table_cell_style),
                    Paragraph(f"{row.get('db*', 0):.1f}", table_cell_style),
                    Paragraph(f"<b>{row.get('delta_e', 0):.2f}</b>", table_cell_style),
                    Paragraph(conc_val, table_cell_style),
                ]
            )

        col_widths = [58, 62, 60, 36, 36, 36, 36, 36, 36, 46, 42]
        data_table = Table(table_rows, colWidths=col_widths, repeatRows=1)
        data_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(data_table)

    story.append(Spacer(1, 12))

    # Section 4: Captured Images & ROI Detail Gallery
    gallery_items = []

    # Include blank reference image if thumbnail exists
    if blank_ref and "annotated_thumb" in blank_ref and blank_ref["annotated_thumb"]:
        gallery_items.append(
            {
                "title": "Blank Reference Baseline",
                "filename": blank_ref.get("image_name", "blank_image"),
                "roi_str": f"x={blank_ref.get('roi',[0,0,0,0])[0]}, y={blank_ref.get('roi',[0,0,0,0])[1]}, w={blank_ref.get('roi',[0,0,0,0])[2]}, h={blank_ref.get('roi',[0,0,0,0])[3]}",
                "rgb_str": blank_rgb_str,
                "de_str": "0.00 (Baseline)",
                "annotated_thumb": blank_ref["annotated_thumb"],
                "spot_thumb": blank_ref.get("spot_thumb", None),
                "resolution": f"{blank_ref.get('image_shape', ['?','?'])[1]}x{blank_ref.get('image_shape', ['?','?'])[0]}" if "image_shape" in blank_ref else "Original",
            }
        )

    # Include sample images
    if not results_df.empty:
        for _, row in results_df.iterrows():
            if "annotated_thumb" in row and row["annotated_thumb"]:
                res_str = f"{row.get('image_shape', ['?','?'])[1]}x{row.get('image_shape', ['?','?'])[0]}" if "image_shape" in row and isinstance(row["image_shape"], (list, tuple)) else "Original"
                gallery_items.append(
                    {
                        "title": str(row.get("sample_name", "Sample")),
                        "filename": str(row.get("image_name", "")),
                        "roi_str": f"x={row.get('roi_x',0)}, y={row.get('roi_y',0)}, w={row.get('roi_w',0)}, h={row.get('roi_h',0)}",
                        "rgb_str": f"({row.get('R',0):.1f}, {row.get('G',0):.1f}, {row.get('B',0):.1f})",
                        "de_str": f"{row.get('delta_e', 0):.2f}",
                        "annotated_thumb": row["annotated_thumb"],
                        "spot_thumb": row.get("spot_thumb", None),
                        "resolution": res_str,
                    }
                )

    if gallery_items:
        story.append(Paragraph("4. Captured Sensor Images & ROI Identification", h2_style))

        gallery_table_rows = []
        for item in gallery_items:
            # Full annotated image
            img_buf = io.BytesIO(base64.b64decode(item["annotated_thumb"]))
            annotated_img_flow = RLImage(img_buf, width=2.0 * inch, height=1.5 * inch)

            # Zoomed spot crop (if available)
            if item["spot_thumb"]:
                spot_buf = io.BytesIO(base64.b64decode(item["spot_thumb"]))
                spot_flow = RLImage(spot_buf, width=1.0 * inch, height=1.0 * inch)
            else:
                spot_flow = Paragraph("No Crop", body_style)

            meta_text = f"""<b>{item['title']}</b><br/>
            <b>File:</b> {item['filename']} ({item['resolution']})<br/>
            <b>ROI Box:</b> {item['roi_str']}<br/>
            <b>Mean RGB:</b> {item['rgb_str']}<br/>
            <b>Measured ΔE:</b> {item['de_str']}
            """
            meta_flow = Paragraph(meta_text, body_style)

            gallery_table_rows.append([annotated_img_flow, spot_flow, meta_flow])

        gallery_table = Table(gallery_table_rows, colWidths=[150, 85, 305])
        gallery_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                    ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(KeepTogether([gallery_table]))

    story.append(Spacer(1, 14))
    story.append(
        Paragraph(
            "<i>Report certified by ChemoSense Image-Based Analytical System &bull; Green boxes delineate the analyzed Regions of Interest (ROIs).</i>",
            subtitle_style,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_html_report(
    results_df: pd.DataFrame,
    blank_ref: Optional[Dict[str, Any]],
    chart_fig: Optional[plt.Figure] = None,
    report_title: str = "ChemoSense Analytical Report",
) -> str:
    """Generates an HTML report with embedded styles, responsive layout, base64 chart, captured images gallery, and print capability."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    blank_rgb_str = "N/A"
    blank_roi_str = "N/A"
    blank_file_str = "N/A"
    if blank_ref:
        b_rgb = blank_ref.get("rgb", [0, 0, 0])
        b_roi = blank_ref.get("roi", [0, 0, 0, 0])
        blank_rgb_str = f"({b_rgb[0]:.1f}, {b_rgb[1]:.1f}, {b_rgb[2]:.1f})"
        blank_roi_str = f"x={b_roi[0]}, y={b_roi[1]}, w={b_roi[2]}, h={b_roi[3]}"
        blank_file_str = str(blank_ref.get("image_name", "blank_image"))

    mean_de = f"{results_df['delta_e'].mean():.2f}" if not results_df.empty else "N/A"
    max_de = f"{results_df['delta_e'].max():.2f}" if not results_df.empty else "N/A"
    min_de = f"{results_df['delta_e'].min():.2f}" if not results_df.empty else "N/A"

    chart_base64 = ""
    if chart_fig is not None:
        img_bytes = fig_to_bytes(chart_fig)
        chart_base64 = base64.b64encode(img_bytes).decode("utf-8")

    rows_html = ""
    if not results_df.empty:
        for _, row in results_df.iterrows():
            rgb_text = f"({row['R']:.1f}, {row['G']:.1f}, {row['B']:.1f})"
            conc_val = f"{row['concentration']:.2f}" if not pd.isna(row.get("concentration")) else "-"
            rows_html += f"""
            <tr>
                <td><strong>{row.get('sample_name', '')}</strong></td>
                <td>{row.get('image_name', '')}</td>
                <td>{rgb_text}</td>
                <td>{row.get('L*', 0):.2f}</td>
                <td>{row.get('a*', 0):.2f}</td>
                <td>{row.get('b*', 0):.2f}</td>
                <td>{row.get('dL*', 0):.2f}</td>
                <td>{row.get('da*', 0):.2f}</td>
                <td>{row.get('db*', 0):.2f}</td>
                <td class="highlight">{row.get('delta_e', 0):.2f}</td>
                <td>{conc_val}</td>
            </tr>
            """

    # Captured images cards HTML
    gallery_cards_html = ""
    if blank_ref and "annotated_thumb" in blank_ref and blank_ref["annotated_thumb"]:
        res_str = f"{blank_ref.get('image_shape', ['?','?'])[1]} x {blank_ref.get('image_shape', ['?','?'])[0]}" if "image_shape" in blank_ref else "Original"
        spot_crop_img = f"<img class='spot-crop' src='data:image/jpeg;base64,{blank_ref['spot_thumb']}' alt='Spot Crop' />" if blank_ref.get("spot_thumb") else ""
        gallery_cards_html += f"""
        <div class="gallery-card">
            <div class="gallery-header">
                <span class="badge badge-blank">Blank Reference</span>
                <span class="filename">{blank_file_str}</span>
            </div>
            <div class="gallery-images">
                <img class="annotated-img" src="data:image/jpeg;base64,{blank_ref['annotated_thumb']}" alt="Blank Sensor" />
                {spot_crop_img}
            </div>
            <div class="gallery-meta">
                <div><strong>Resolution:</strong> {res_str}</div>
                <div><strong>ROI Box:</strong> {blank_roi_str}</div>
                <div><strong>Mean RGB:</strong> {blank_rgb_str}</div>
                <div><strong>ΔE:</strong> 0.00 (Reference)</div>
            </div>
        </div>
        """

    if not results_df.empty:
        for _, row in results_df.iterrows():
            if "annotated_thumb" in row and row["annotated_thumb"]:
                res_str = f"{row.get('image_shape', ['?','?'])[1]} x {row.get('image_shape', ['?','?'])[0]}" if "image_shape" in row and isinstance(row["image_shape"], (list, tuple)) else "Original"
                spot_crop_img = f"<img class='spot-crop' src='data:image/jpeg;base64,{row['spot_thumb']}' alt='Spot Crop' />" if row.get("spot_thumb") else ""
                gallery_cards_html += f"""
                <div class="gallery-card">
                    <div class="gallery-header">
                        <span class="badge badge-sample">{row.get('sample_name', 'Sample')}</span>
                        <span class="filename">{row.get('image_name', '')}</span>
                    </div>
                    <div class="gallery-images">
                        <img class="annotated-img" src="data:image/jpeg;base64,{row['annotated_thumb']}" alt="{row.get('sample_name', 'Sample')}" />
                        {spot_crop_img}
                    </div>
                    <div class="gallery-meta">
                        <div><strong>Resolution:</strong> {res_str}</div>
                        <div><strong>ROI Box:</strong> x={row.get('roi_x',0)}, y={row.get('roi_y',0)}, w={row.get('roi_w',0)}, h={row.get('roi_h',0)}</div>
                        <div><strong>Mean RGB:</strong> ({row.get('R',0):.1f}, {row.get('G',0):.1f}, {row.get('B',0):.1f})</div>
                        <div><strong>Measured ΔE:</strong> <span class="de-val">{row.get('delta_e', 0):.2f}</span></div>
                    </div>
                </div>
                """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{report_title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 30px auto;
            max-width: 980px;
            color: #1e293b;
            background-color: #ffffff;
            line-height: 1.5;
            padding: 0 20px;
        }}
        .header {{
            border-bottom: 2px solid #2563eb;
            padding-bottom: 12px;
            margin-bottom: 24px;
        }}
        .header h1 {{
            margin: 0 0 6px 0;
            color: #1e3a8a;
            font-size: 26px;
        }}
        .header .meta {{
            color: #64748b;
            font-size: 13px;
        }}
        .section-title {{
            color: #1e3a8a;
            font-size: 18px;
            margin-top: 28px;
            margin-bottom: 12px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4px;
        }}
        .card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }}
        .card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px 16px;
        }}
        .card .label {{
            font-size: 12px;
            color: #64748b;
            text-transform: uppercase;
            font-weight: 600;
        }}
        .card .value {{
            font-size: 18px;
            font-weight: 700;
            color: #0f172a;
            margin-top: 4px;
        }}
        .chart-box {{
            text-align: center;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 24px;
        }}
        .chart-box img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 8px;
            background: #ffffff;
        }}
        th {{
            background-color: #1e3a8a;
            color: #ffffff;
            text-align: left;
            padding: 9px 10px;
            font-weight: 600;
        }}
        td {{
            padding: 8px 10px;
            border-bottom: 1px solid #e2e8f0;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        td.highlight {{
            font-weight: 700;
            color: #166534;
        }}
        .gallery-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 18px;
            margin-top: 14px;
        }}
        .gallery-card {{
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 12px;
            display: flex;
            flex-direction: column;
        }}
        .gallery-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .badge {{
            font-size: 11px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }}
        .badge-blank {{
            background: #e2e8f0;
            color: #334155;
        }}
        .badge-sample {{
            background: #dbeafe;
            color: #1e40af;
        }}
        .filename {{
            font-size: 11px;
            color: #64748b;
            font-family: monospace;
        }}
        .gallery-images {{
            display: flex;
            gap: 8px;
            align-items: center;
            justify-content: center;
            background: #0f172a;
            border-radius: 6px;
            padding: 8px;
            min-height: 140px;
        }}
        .annotated-img {{
            max-height: 130px;
            max-width: 65%;
            object-fit: contain;
            border-radius: 4px;
        }}
        .spot-crop {{
            max-height: 100px;
            width: auto;
            border-radius: 50%;
            border: 2px solid #22c55e;
            box-shadow: 0 0 6px rgba(0,0,0,0.5);
        }}
        .gallery-meta {{
            font-size: 12px;
            color: #334155;
            margin-top: 10px;
            line-height: 1.6;
        }}
        .de-val {{
            font-weight: 700;
            color: #166534;
        }}
        .footer {{
            margin-top: 36px;
            padding-top: 14px;
            border-top: 1px solid #e2e8f0;
            font-size: 12px;
            color: #94a3b8;
            text-align: center;
        }}
        @media print {{
            body {{
                margin: 0;
                padding: 0;
                max-width: 100%;
            }}
            .gallery-container {{
                grid-template-columns: 1fr 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_title}</h1>
        <div class="meta">Generated: {now_str} &bull; Pipeline: CIELAB ΔE (CIE76 / D65)</div>
    </div>

    <div class="section-title">1. Baseline & Summary Statistics</div>
    <div class="card-grid">
        <div class="card">
            <div class="label">Blank RGB</div>
            <div class="value">{blank_rgb_str}</div>
        </div>
        <div class="card">
            <div class="label">Total Samples</div>
            <div class="value">{len(results_df)}</div>
        </div>
        <div class="card">
            <div class="label">Mean ΔE</div>
            <div class="value">{mean_de}</div>
        </div>
        <div class="card">
            <div class="label">ΔE Range (Min / Max)</div>
            <div class="value">{min_de} / {max_de}</div>
        </div>
    </div>

    {"<div class='section-title'>2. Colorimetric Response Visualization</div><div class='chart-box'><img src='data:image/png;base64," + chart_base64 + "' alt='Delta E Chart' /></div>" if chart_base64 else ""}

    <div class="section-title">3. Quantitative Results Table</div>
    <table>
        <thead>
            <tr>
                <th>Sample</th>
                <th>Image</th>
                <th>Mean RGB</th>
                <th>L*</th>
                <th>a*</th>
                <th>b*</th>
                <th>ΔL*</th>
                <th>Δa*</th>
                <th>Δb*</th>
                <th>ΔE</th>
                <th>Conc.</th>
            </tr>
        </thead>
        <tbody>
            {rows_html if rows_html else "<tr><td colspan='11'>No sample data recorded.</td></tr>"}
        </tbody>
    </table>

    {"<div class='section-title'>4. Captured Sensor Images & ROI Identification</div><div class='gallery-container'>" + gallery_cards_html + "</div>" if gallery_cards_html else ""}

    <div class="footer">
        Generated by ChemoSense Analytical System &bull; Transduction: CIELAB Euclidean Metric (CIE76)
    </div>
</body>
</html>
"""
    return html
