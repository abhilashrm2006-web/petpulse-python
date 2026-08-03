"""Covers is_tool_allowed_for_role -- the only gating layer left in the
registry now that every feature except a paid doctor consultation is free
for every customer (there's no more Subscriber-vs-Free tier gate)."""

from app.agent.registry import is_tool_allowed_for_role


def test_customer_tools_open_to_customers():
    for name in ("book_slot", "search_documents", "get_shareable_link", "request_doctor_session", "select_doctor"):
        assert is_tool_allowed_for_role(name, "customer") is True


def test_vet_only_tools_blocked_for_customers():
    for name in ("accept_session", "decline_session", "mark_session_done", "file_prescription", "list_my_appointments"):
        assert is_tool_allowed_for_role(name, "customer") is False
        assert is_tool_allowed_for_role(name, "vet") is True


def test_data_deletion_tools_are_customer_only():
    for name in ("request_data_deletion", "respond_to_deletion_confirmation", "record_deletion_feedback"):
        assert is_tool_allowed_for_role(name, "customer") is True
        assert is_tool_allowed_for_role(name, "vet") is False


def test_unknown_tool_name_is_never_allowed():
    assert is_tool_allowed_for_role("not_a_real_tool", "customer") is False
