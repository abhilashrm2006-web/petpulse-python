-- Remaining items from PetPulse_Bot_Changes_Spec.docx after the 2026-08-04
-- reengagement/onboarding-instrumentation workstream shipped (see
-- migrations/2026_08_04_reengagement_workstream.sql). Idempotent
-- (IF NOT EXISTS throughout) so it's safe to re-run.

-- 1. Onboarding funnel visibility (item #2 remainder): append-only audit
-- trail of every registration_step transition, so time-in-step can be
-- measured directly instead of inferred from last_active_at being blank.
CREATE TABLE IF NOT EXISTS registration_step_history (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id uuid NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    from_step text,
    to_step text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS registration_step_history_profile_id_idx ON registration_step_history(profile_id);

-- 2. Nearby-vet tool resilience (item #1): admin-curated fallback directory,
-- checked only when the live Overpass/OSM lookup fails after retries or
-- returns nothing for the customer's area. Starts empty -- ops seeds real,
-- verified clinics per city; the app never fabricates entries here.
CREATE TABLE IF NOT EXISTS vet_directory_fallback (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    city text NOT NULL,
    name text NOT NULL,
    address text,
    phone text,
    website text,
    opening_hours text,
    maps_url text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS vet_directory_fallback_city_idx ON vet_directory_fallback(lower(city));

-- 3. Paywall sequencing (item #3): dedicated cooldown column for the
-- automatic price_objection_silence follow-up nudge, kept separate from
-- last_reengagement_sent_at (the unrelated 48h gone-quiet job) and
-- onboarding_migration_nudge_sent_at (the unrelated stranded-signup
-- broadcast).
ALTER TABLE profiles
    ADD COLUMN IF NOT EXISTS last_price_objection_nudge_sent_at timestamptz;
