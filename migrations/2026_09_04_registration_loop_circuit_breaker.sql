-- Registration wizard circuit breaker (2026-09-04)
-- Root cause: a business WhatsApp auto-responder ("Asset Tree Homes") got
-- caught in the registration wizard and bounced the identical canned reply
-- back at us 187 times -- our re-prompt triggered their auto-reply, which
-- triggered our next re-prompt, forever. Idempotent (IF NOT EXISTS), safe
-- to re-run.

ALTER TABLE profiles
    ADD COLUMN IF NOT EXISTS last_rejected_registration_input text,
    ADD COLUMN IF NOT EXISTS repeated_rejection_count integer NOT NULL DEFAULT 0;
