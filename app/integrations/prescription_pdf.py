"""Renders a filed prescription as a downloadable, letterhead-branded PDF
using fpdf2 -- pure Python, no native/system libraries (unlike e.g.
WeasyPrint), so it adds nothing to the Docker image beyond a small pip
package. This is a rendering step only: the actual clinical content
(medications/treatment plan) still comes verbatim from file_prescription's
already-formatted message, never regenerated or altered here.

The clinic block (logo, address, phone, email) is fixed -- Waggy World's
own letterhead identity. The doctor block (name, qualification, registration
number, contact) is per-consultation, sourced from the treating vet's own
profile row."""

import os

from fpdf import FPDF

PAGE_MARGIN_MM = 18
LOGO_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "waggy_world_logo.png")

CLINIC_NAME = "Waggy World"
CLINIC_ADDRESS = "Plot No. 13, Beside Agarwal Homes, Balaji Nagar 3rd Extension, Puzhuthivakkam, Chennai - 600091, Tamil Nadu"
CLINIC_PHONE = "+91 9513808305"
CLINIC_EMAIL = "abhilash@WaggyWorld1993.onmicrosoft.com"

# fpdf2's core fonts (Helvetica etc.) only support latin-1 -- embedding a
# full Unicode TTF just for this would mean bundling a font file. Since this
# renders free-typed vet input (which, live, has already included stray
# smart-quotes/dashes and even non-Latin text elsewhere in this system),
# normalize common typographic characters to their ASCII equivalents first,
# then let anything truly unencodable degrade to "?" rather than crash the
# whole PDF -- the WhatsApp text message already sent alongside it is the
# full-fidelity, full-Unicode record either way.
_TYPOGRAPHIC_REPLACEMENTS = {
    "—": "-", "–": "-",  # em/en dash
    "‘": "'", "’": "'",  # curly single quotes
    "“": '"', "”": '"',  # curly double quotes
    "…": "...",  # ellipsis
    " ": " ",  # non-breaking space
}


def _pdf_safe(text: str) -> str:
    for bad, good in _TYPOGRAPHIC_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_prescription_pdf(
    *,
    pet_name: str,
    doctor_name: str,
    doctor_qualification: str = "",
    doctor_registration_number: str = "",
    doctor_phone: str = "",
    date_str: str,
    reason: str,
    medications: str,
    treatment_plan: str,
) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_margin(PAGE_MARGIN_MM)
    pdf.add_page()

    # --- clinic header: logo left, contact block right ---
    page_w = pdf.w
    logo_w = 48
    if os.path.exists(LOGO_PATH):
        pdf.image(LOGO_PATH, x=PAGE_MARGIN_MM, y=PAGE_MARGIN_MM, w=logo_w)

    pdf.set_xy(page_w / 2, PAGE_MARGIN_MM)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(90, 90, 90)
    contact_w = page_w / 2 - PAGE_MARGIN_MM
    pdf.multi_cell(contact_w, 4.2, _pdf_safe(CLINIC_ADDRESS), align="R")
    pdf.set_x(page_w / 2)
    pdf.cell(contact_w, 4.2, _pdf_safe(CLINIC_PHONE), align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(page_w / 2)
    pdf.cell(contact_w, 4.2, _pdf_safe(CLINIC_EMAIL), align="R", new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(max(pdf.get_y(), PAGE_MARGIN_MM + 20) + 4)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(PAGE_MARGIN_MM, pdf.get_y(), page_w - PAGE_MARGIN_MM, pdf.get_y())
    pdf.ln(6)

    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 8, "Prescription", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # --- doctor block (editable per treating vet) ---
    pdf.set_font("Helvetica", "B", 11.5)
    pdf.cell(0, 6.5, _pdf_safe(f"Dr. {doctor_name}" if not doctor_name.lower().startswith("dr") else doctor_name), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    if doctor_qualification:
        pdf.cell(0, 5.5, _pdf_safe(doctor_qualification), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5.5, _pdf_safe(f"Registration No: {doctor_registration_number or 'Not on file'}"), new_x="LMARGIN", new_y="NEXT")
    if doctor_phone:
        pdf.cell(0, 5.5, _pdf_safe(f"Contact: {doctor_phone}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- patient / date ---
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, _pdf_safe(f"Patient: {pet_name}"), new_x="LMARGIN", new_y="NEXT")
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
