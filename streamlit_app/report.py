"""Export helpers: JSON dump and a one-page PDF summary of a completed analysis."""

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any

from fpdf import FPDF

DISCLAIMER = "Research prototype. Not intended, validated, or suitable for clinical use."
GRADCAM_CAVEAT = (
    "Grad-CAM highlights image regions that influenced the model's prediction. "
    "It reflects model attention, not a confirmed or radiologist-verified disease location."
)


def build_json_report(result: dict[str, Any], model_info: dict[str, Any], filename: str) -> bytes:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "filename": filename,
        "disclaimer": DISCLAIMER,
        "model": model_info,
        "result": result,
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def build_pdf_report(
    result: dict[str, Any],
    model_info: dict[str, Any],
    filename: str,
    xray_bytes: bytes,
    gradcam_bytes: bytes | None = None,
    gradcam_pathology: str | None = None,
) -> bytes:
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(26, 35, 50)
    pdf.cell(0, 10, "Chest X-Ray AI Analysis Report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(90, 100, 120)
    pdf.multi_cell(0, 5, DISCLAIMER)
    pdf.ln(1)

    pdf.set_draw_color(220, 224, 230)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(26, 35, 50)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    auc = model_info.get("val_macro_auc")
    pdf.cell(0, 6, f"File: {filename}   |   Generated: {generated}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(
        0, 6,
        f"Model: {model_info.get('model_version', '-')}  "
        f"(epoch {model_info.get('checkpoint_epoch', '-')}, "
        f"val AUROC {f'{auc:.4f}' if auc else '-'})",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(3)

    img_y = pdf.get_y()
    pdf.image(io.BytesIO(xray_bytes), x=10, y=img_y, w=90)
    caption = "Original X-ray"
    if gradcam_bytes:
        pdf.image(io.BytesIO(gradcam_bytes), x=110, y=img_y, w=90)
        caption = f"Original X-ray{' ' * 60}Grad-CAM: {gradcam_pathology}"
    pdf.set_y(img_y + 95)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(90, 100, 120)
    pdf.cell(0, 5, caption, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(26, 35, 50)
    pdf.cell(0, 7, "Findings", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(244, 246, 249)
    pdf.cell(70, 7, "Pathology", border=1, fill=True)
    pdf.cell(40, 7, "Probability", border=1, fill=True)
    pdf.cell(40, 7, "Threshold", border=1, fill=True)
    pdf.cell(40, 7, "Above threshold", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    for p in result["predictions"]:
        if p["above_threshold"]:
            pdf.set_text_color(180, 83, 9)
            pdf.set_font("Helvetica", "B", 9)
        else:
            pdf.set_text_color(60, 70, 90)
            pdf.set_font("Helvetica", "", 9)
        pdf.cell(70, 6.5, p["pathology"], border=1)
        pdf.cell(40, 6.5, f"{p['probability']:.1%}", border=1)
        pdf.cell(40, 6.5, f"{p['threshold']:.2f}", border=1)
        pdf.cell(40, 6.5, "Yes" if p["above_threshold"] else "No", border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(90, 100, 120)
    pdf.multi_cell(0, 5, GRADCAM_CAVEAT)

    return bytes(pdf.output())
