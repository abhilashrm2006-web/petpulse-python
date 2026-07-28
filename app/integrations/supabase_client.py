"""Supabase client + a handful of thin, reused repo helpers.

Uses the official `supabase-py` client (service-role key) against the exact
same 24-table schema the n8n workflows use — no migration. The
`pets?select=*,pet_members!inner()&pet_members.profile_id=eq.{id}` join
pattern (spec §2, duplicated raw-REST in 5+ n8n workflows) is the real
multi-owner access-control mechanism throughout the system; it's
consolidated here as `get_pets_for_profile` instead of being re-typed at
every call site.
"""

from typing import Any

import httpx
from supabase import Client, create_client

from app.config import Settings


def make_supabase_client(settings: Settings) -> Client:
    client = create_client(settings.supabase_url, settings.supabase_service_role_key)
    # Confirmed live: a pooled HTTP/2 keep-alive connection that Supabase's
    # side closed got reused anyway and raised httpcore.RemoteProtocolError
    # ("Server disconnected") — httpx's default transport does zero retries,
    # so that took down an entire inbound WhatsApp turn (customer got no
    # reply at all) for a transient blip that a fresh connection would have
    # sailed through. Retrying is safe here: these are connection failures
    # before any response was received, never a retry of a completed write.
    retrying_transport = httpx.HTTPTransport(retries=2)
    client.postgrest.session._transport = retrying_transport
    client.storage._client._transport = retrying_transport
    return client


def get_pets_for_profile(client: Client, profile_id: str) -> list[dict[str, Any]]:
    """A pet is visible to a profile iff a `pet_members` row links them —
    this is the actual access-control boundary (owner, family, caregiver,
    vet), not `pets.profile_id` alone."""
    resp = (
        client.table("pets")
        .select("*, pet_members!inner(profile_id, role, is_primary)")
        .eq("pet_members.profile_id", profile_id)
        .eq("is_active", True)
        .order("created_at")
        .execute()
    )
    pets = resp.data or []
    _attach_owner_info(client, pets)
    return pets


def _attach_owner_info(client: Client, pets: list[dict[str, Any]]) -> None:
    """The `pet_members` row embedded above is only the CALLER's own
    membership on each pet — for a vet, whose patient list spans every
    unrelated household that's added them, that says nothing about whose
    pet it actually is. Attaches each pet's real primary owner (name +
    phone) so the agent can tell two same-named pets apart using
    conversation context (e.g. an owner's name the vet mentioned) instead
    of a name-only guess — a real reported bug sent one owner's document
    to a different owner's same-named pet."""
    pet_ids = [p["id"] for p in pets if p.get("id")]
    if not pet_ids:
        return
    resp = (
        client.table("pet_members")
        .select("pet_id, profiles!pet_members_profile_id_fkey(full_name, phone_number)")
        .in_("pet_id", pet_ids)
        .eq("is_primary", True)
        .execute()
    )
    owners = {row["pet_id"]: row.get("profiles") or {} for row in (resp.data or [])}
    for pet in pets:
        owner = owners.get(pet.get("id")) or {}
        pet["owner_name"] = owner.get("full_name")
        pet["owner_phone"] = owner.get("phone_number")


def get_pet_member_contacts(client: Client, pet_id: str) -> list[dict[str, Any]]:
    """Every person (owner/family/caregiver) linked to a pet — phone_number,
    full_name, email — used to broadcast session notifications to the whole
    household, not just whichever member happens to be driving the current
    WhatsApp conversation. `pet_members` has two FKs into `profiles`
    (profile_id and added_by), so the embed must be disambiguated by FK
    name or PostgREST rejects it as ambiguous (confirmed live)."""
    resp = (
        client.table("pet_members")
        .select("profile_id, role, profiles!pet_members_profile_id_fkey(phone_number, full_name, email)")
        .eq("pet_id", pet_id)
        .execute()
    )
    contacts = []
    for row in resp.data or []:
        profile = row.get("profiles") or {}
        if profile.get("phone_number"):
            contacts.append({"profile_id": row["profile_id"], "role": row.get("role"), **profile})
    return contacts


def get_profile_by_phone(client: Client, phone_number: str) -> dict[str, Any] | None:
    resp = client.table("profiles").select("*").eq("phone_number", phone_number).limit(1).execute()
    return resp.data[0] if resp.data else None


def is_active_subscriber(client: Client, profile_id: str) -> bool:
    """True only once a subscription is actually confirmed by Razorpay's
    webhook (status="active") -- status="trial" is set the moment someone
    taps Subscribe, before they've completed payment authorization, and
    deliberately does not grant access on its own."""
    rows = client.table("subscriptions").select("id").eq("profile_id", profile_id).eq("status", "active").limit(1).execute().data
    return bool(rows)


def get_or_create_profile(client: Client, phone_number: str, sender_name: str) -> dict[str, Any]:
    existing = get_profile_by_phone(client, phone_number)
    if existing:
        return existing
    resp = (
        client.table("profiles")
        .insert({"phone_number": phone_number, "full_name": sender_name, "role": "customer"})
        .execute()
    )
    return resp.data[0]


def escape_or_filter_value(value: str) -> str:
    """PostgREST's `.or_("col.op.value,col2.op.value2")` filter string treats
    a bare comma or parenthesis in `value` as filter syntax (a new clause or
    a group), not literal text -- an unescaped user-supplied value (e.g. an
    admin search box, or free-text message content used as an ilike pattern)
    can inject extra clauses into the query. Per PostgREST's documented
    escaping, wrapping the value in double quotes makes it literal; inside
    those quotes, a literal backslash or double quote must itself be
    backslash-escaped."""
    if any(c in value for c in ',()"\\'):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def is_unique_violation(exc: Exception) -> bool:
    """postgrest raises a generic APIError on any DB error — this is the
    only way to distinguish a real unique-constraint conflict (23505) from
    every other failure mode, which must still propagate."""
    return "23505" in str(exc) or "duplicate key" in str(exc).lower()


def claim_message_id(client: Client, message_id: str) -> bool:
    """Dedup: attempt to INSERT into `processed_messages` (PK=message_id).
    Returns False (already processed) on a unique-violation, True on a
    fresh claim. Mirrors n8n's `Claim Message Id` -> `Duplicate - Skip`."""
    try:
        client.table("processed_messages").insert({"message_id": message_id}).execute()
        return True
    except Exception as exc:
        if is_unique_violation(exc):
            return False
        raise


def sign_storage_url(client: Client, bucket: str, object_path: str, expires_in: int = 3600) -> str:
    result = client.storage.from_(bucket).create_signed_url(object_path, expires_in)
    return result["signedURL"] if "signedURL" in result else result.get("signedUrl", "")


def upload_to_storage(client: Client, bucket: str, object_path: str, data: bytes, mime_type: str) -> None:
    client.storage.from_(bucket).upload(
        object_path, data, file_options={"content-type": mime_type, "upsert": "true"}
    )
