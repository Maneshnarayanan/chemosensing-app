import io
import base64
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
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


def generate_pdf_report(
    results_df: pd.DataFrame,
    blank_ref: Optional[Dict[str, Any]],
    chart_fig: Optional[plt.Figure] = None,
    report_title: str = "ChemoSense Analytical Report",
) -> bytes:
    """Generates a comprehensive PDF report containing metadata, blank ref, data table, and graph."""
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
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0F172A"),
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
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
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=12))

    # Section 1: Overview & Blank Reference
    story.append(Paragraph("1. Session Overview & Calibration Baseline", h2_style))

    blank_rgb_str = "N/A"
    blank_roi_str = "N/A"
    if blank_ref:
        b_rgb = blank_ref.get("rgb", [0, 0, 0])
        b_roi = blank_ref.get("roi", [0, 0, 0, 0])
        blank_rgb_str = f"({b_rgb[0]:.1f}, {b_rgb[1]:.1f}, {b_rgb[2]:.1f})"
        blank_roi_str = f"x={b_roi[0]}, y={b_roi[1]}, w={b_roi[2]}, h={b_roi[3]}"

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
            Paragraph("<b>Blank ROI Bounding Box:</b>", body_style),
            Paragraph(blank_roi_str, body_style),
            Paragraph("<b>Mean ΔE:</b>", body_style),
            Paragraph(mean_de, body_style),
        ],
        [
            Paragraph("<b>Color Space / Illuminant:</b>", body_style),
            Paragraph("CIELAB (D65 Standard)", body_style),
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
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # Section 2: Visualizations (Chart)
    if chart_fig is not None:
        story.append(Paragraph("2. Quantitative Colorimetric Response (ΔE Chart)", h2_style))
        img_data = fig_to_bytes(chart_fig)
        img_buf = io.BytesIO(img_data)
        chart_img = RLImage(img_buf, width=6.8 * inch, height=3.2 * inch)
        story.append(chart_img)
        story.append(Spacer(1, 14))

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
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(data_table)

    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            "<i>Note: ΔE values indicate color difference from the baseline reference. Higher ΔE represents greater chromatic deviation corresponding to analyte presence.</i>",
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
    """Generates an HTML report with embedded styles, responsive layout, base64 chart, and print capability."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    blank_rgb_str = "N/A"
    blank_roi_str = "N/A"
    if blank_ref:
        b_rgb = blank_ref.get("rgb", [0, 0, 0])
        b_roi = blank_ref.get("roi", [0, 0, 0, 0])
        blank_rgb_str = f"({b_rgb[0]:.1f}, {b_rgb[1]:.1f}, {b_rgb[2]:.1f})"
        blank_roi_str = f"x={b_roi[0]}, y={b_roi[1]}, w={b_roi[2]}, h={b_roi[3]}"

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

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{report_title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 30px auto;
            max-width: 950px;
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
            margin-top: 24px;
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
            .no-print {{
                display: none;
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

    <div class="footer">
        Generated by ChemoSense Analytical System &bull; Transduction: CIELAB Euclidean Metric (CIE76)
    </div>
</body>
</html>
"""
    return html
