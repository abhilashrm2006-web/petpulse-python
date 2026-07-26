"""Covers the deterministic, safety-critical pieces of check_symptoms that
are deliberately NOT left to LLM judgement: the severity display string
(so the rating always reads the same regardless of phrasing) and the
keyword-based severity escalation (red-flag terms, and the ingestion x
harmful-substance intersection for poisoning claims)."""

from app.agent.tools.symptoms import _apply_safety_override, _severity_display


def test_severity_display_mild_is_green():
    assert _severity_display(1, "Mild") == "🟢 Mild (1/5)"


def test_severity_display_moderate_is_yellow():
    assert _severity_display(3, "Moderate") == "🟡 Moderate (3/5)"


def test_severity_display_serious_is_orange():
    assert _severity_display(4, "Serious") == "🟠 Serious (4/5)"


def test_severity_display_emergency_is_red():
    assert _severity_display(5, "Emergency") == "🔴 Emergency (5/5)"


def test_red_flag_keyword_forces_severity_5():
    verdict = {"severity": 2, "severity_label": "Mild", "requires_emergency_care": False}
    result = _apply_safety_override("he can't breathe and his gums look blue", verdict)
    assert result["severity"] == 5
    assert result["severity_label"] == "Emergency"
    assert result["requires_emergency_care"] is True


def test_poisoning_intersection_escalates():
    verdict = {"severity": 1, "severity_label": "Mild", "requires_emergency_care": False}
    result = _apply_safety_override("my dog ate a whole bar of chocolate an hour ago", verdict)
    assert result["severity"] == 5
    assert result["requires_emergency_care"] is True
    assert any("chocolate" in flag for flag in result["red_flags"])


def test_ingestion_without_harmful_substance_does_not_escalate():
    verdict = {"severity": 2, "severity_label": "Mild", "requires_emergency_care": False}
    result = _apply_safety_override("he ate his brother's kibble again", verdict)
    assert result["severity"] == 2
    assert result["requires_emergency_care"] is False


def test_no_red_flags_leaves_verdict_untouched():
    verdict = {"severity": 2, "severity_label": "Mild", "requires_emergency_care": False}
    result = _apply_safety_override("slightly less energetic than usual today", verdict)
    assert result["severity"] == 2
    assert result["requires_emergency_care"] is False


def test_poisoning_intersection_escalates_even_when_llm_sent_explicit_null_red_flags():
    """Audit bug: verdict.setdefault("red_flags", []) only fills in a
    MISSING key -- an LLM emitting a present-but-null "red_flags" (valid
    JSON, plausible when it has nothing to report on its own) left it None,
    and .append() on None crashed exactly on this poisoning/ingestion path,
    the one case this override exists to protect."""
    verdict = {"severity": 1, "severity_label": "Mild", "requires_emergency_care": False, "red_flags": None}
    result = _apply_safety_override("my dog ate a whole bar of chocolate an hour ago", verdict)
    assert result["severity"] == 5
    assert result["requires_emergency_care"] is True
    assert any("chocolate" in flag for flag in result["red_flags"])
