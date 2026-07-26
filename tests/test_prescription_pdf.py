from app.integrations.prescription_pdf import build_prescription_pdf


def test_produces_valid_pdf_bytes():
    pdf_bytes = build_prescription_pdf(
        pet_name="Bobby",
        doctor_name="Dr. Rao",
        date_str="26 Jul 2026",
        reason="Coughing for 2 days",
        medications="Amoxicillin 250mg twice daily",
        treatment_plan="Rest for 5 days",
    )
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 200


def test_survives_typographic_and_non_latin1_characters():
    """A vet's own free-typed text has, live, included smart quotes/dashes
    and non-Latin characters — fpdf2's core font only supports Latin-1 and
    raises on anything else, so this must degrade gracefully instead of
    crashing the whole PDF."""
    pdf_bytes = build_prescription_pdf(
        pet_name="Bobby",
        doctor_name="Dr. Rao",
        date_str="26 Jul 2026",
        reason="Owner's note — “needs a recheck” in 5 days… 日本語 test",
        medications="Amoxicillin 250mg — twice daily",
        treatment_plan="",
    )
    assert pdf_bytes.startswith(b"%PDF")


def test_omits_empty_optional_sections_without_crashing():
    pdf_bytes = build_prescription_pdf(
        pet_name="Bobby", doctor_name="Dr. Rao", date_str="26 Jul 2026", reason="", medications="Amoxicillin", treatment_plan=""
    )
    assert pdf_bytes.startswith(b"%PDF")
