-- Re-engagement and onboarding-instrumentation workstream (2026-08-04)
-- Run this once against the Supabase Postgres instance before deploying the
-- app code that references these columns/tables. Idempotent (IF NOT EXISTS
-- throughout) so it's safe to re-run.

-- 1. Stranded pre-migration onboarding broadcast: dedicated column so this
-- nudge's "already sent" state never collides with last_reengagement_sent_at
-- (which belongs to the unrelated 48h gone-quiet job).
ALTER TABLE profiles
    ADD COLUMN IF NOT EXISTS onboarding_migration_nudge_sent_at timestamptz,
    ADD COLUMN IF NOT EXISTS onboarding_migration_nudge_status text;

-- 2. Onboarding drop-off instrumentation: one row per inbound message
-- received while a profile is mid-registration, so we can finally tell
-- "replied and got rejected" apart from "never replied" for the two new-flow
-- steps (awaiting_customer_name, awaiting_pet_name).
CREATE TABLE IF NOT EXISTS onboarding_events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id uuid NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    registration_step text NOT NULL,
    raw_input text,
    validator_result text NOT NULL,  -- 'accepted' | 'rejected'
    rejection_reason text,           -- null when accepted
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS onboarding_events_profile_id_idx ON onboarding_events(profile_id);
CREATE INDEX IF NOT EXISTS onboarding_events_step_result_idx ON onboarding_events(registration_step, validator_result);

-- 3. Emergency-escalation human-checkin queue: flags a health_logs row (a
-- check_symptoms assessment) that came back RED/emergency and the customer
-- never sent a follow-up message within the review window, so a person can
-- do a manual check-in instead of relying on the automated broadcast system.
ALTER TABLE health_logs
    ADD COLUMN IF NOT EXISTS needs_human_checkin boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS human_checkin_flagged_at timestamptz,
    ADD COLUMN IF NOT EXISTS human_checkin_resolved_at timestamptz;

CREATE INDEX IF NOT EXISTS health_logs_needs_checkin_idx ON health_logs(needs_human_checkin) WHERE needs_human_checkin = true;
