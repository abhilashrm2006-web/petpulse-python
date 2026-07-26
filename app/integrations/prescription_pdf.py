"""Renders a filed prescription as a downloadable PDF using fpdf2 — pure
Python, no native/system libraries (unlike e.g. WeasyPrint), so it adds
nothing to the Docker image beyond a small pip package. This is a rendering
step only: the actual clinical content (medications/treatment plan) still
comes verbatim from file_prescription's already-formatted message, never
regenerated or altered here."""

from fpdf import FPDF

PAGE_MARGIN_MM = 18

# fpdf2's core fonts (Helvetica etc.) only support latin-1 -- embedding a
# full Unicode TTF just for this would mean bundling a font file. Since this
# renders free-typed vet input (which, live, has already included stray
# smart-quotes/dashes and even Devanagari text elsewhere in this system),
# normalize common typographic characters to their ASCII equivalents first,
# then let anything truly unencodable degrade to "?" rather than crash the
# whole PDF -- the WhatsApp text message already sent alongside it is the
# full-fidelity, full-Unicode record either way.
_TYPOGRAPHIC_REPLACEMENTS = {
    "—": "-", "–": "-",  # em/en dash
    "‘": "'", "’": "'",  # curly single quotes
    "“": '"', "”": '"',  # curly double quotes
    "…": "...",  # ellipsis
    " ": " ",  # non-breaking space
}


def _pdf_safe(text: str) -> str:
    for bad, good in _TYPOGRAPHIC_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_prescription_pdf(
    *,
    pet_name: str,
    doctor_name: str,
    date_str: str,
    reason: str,
    medications: str,
    treatment_plan: str,
) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_margin(PAGE_MARGIN_MM)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "PetPulse Prescription", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_draw_color(200, 200, 200)
    pdf.line(PAGE_MARGIN_MM, pdf.get_y(), 210 - PAGE_MARGIN_MM, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, _pdf_safe(f"Patient: {pet_name}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, _pdf_safe(f"Veterinarian: {doctor_name}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, _pdf_safe(f"Date: {date_str}"), new_x="LMARGIN", new_y="NEXT")

    if reason:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Reason for Visit", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6.5, _pdf_safe(reason))

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Rx - Medications", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6.5, _pdf_safe(medications or "Not specified"))

    if treatment_plan:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Treatment Plan / Advice", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6.5, _pdf_safe(treatment_plan))

    pdf.ln(12)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, "Generated via PetPulse. This document reflects the veterinarian's own notes from the consultation.")

    return bytes(pdf.output())
