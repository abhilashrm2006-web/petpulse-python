from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # WhatsApp
    whatsapp_access_token: str = ""
    whatsapp_phone_number_id: str = "1186710814528940"
    whatsapp_verify_token: str = "petpulse_wh_9f3a7c2e1b6d4581"
    whatsapp_app_secret: str = ""
    whatsapp_api_version: str = "v21.0"
    # Name of the Meta-approved WhatsApp template used for one-off outreach
    # (e.g. scripts/reengage_stranded_onboarding.py) sent outside the 24h
    # customer-service session window, where free-form text isn't reliably
    # deliverable. Empty until a template is actually created and approved
    # in Meta Business Manager -- scripts that need it must refuse to send
    # live (dry-run only) while this is unset, not silently fall back to
    # free-form text.
    whatsapp_reengagement_template_name: str = ""
    whatsapp_reengagement_template_language: str = "en"
    # Generic single-variable wrapper template (2026-08-27) for every
    # RECURRING proactive/scheduled customer nudge (48h gone-quiet
    # reengagement, 24h stuck-onboarding reminder, price-objection-silence
    # nudge, vaccination reminders, new-parent followups) -- these all
    # compose fully dynamic message text with no fixed shape a 2-3
    # variable template could hold, unlike the one-off broadcast scripts
    # above. See app.integrations.proactive_messaging.send_proactive_message,
    # which picks free-form send_text vs this template automatically based
    # on whether the customer is still inside WhatsApp's 24h customer-service
    # session window -- confirmed live (2026-08-27): every one of these jobs
    # was silently failing delivery (error 131047) for any customer outside
    # that window, since free-form text isn't reliably deliverable there.
    whatsapp_generic_nudge_template_name: str = ""
    whatsapp_generic_nudge_template_language: str = "en"
    # One-off early-user feedback survey broadcast (see
    # scripts/send_feedback_survey.py) -- a fully static template (no
    # variables), submitted 2026-09-01, so this exact wording reaches every
    # customer regardless of their 24h session-window state.
    whatsapp_feedback_survey_template_name: str = ""
    whatsapp_feedback_survey_template_language: str = "en"

    # OpenAI
    openai_api_key: str = ""
    # Full gpt-5.6-sol, not the -mini tier -- the "24/7 AI Health Copilot" is meant
    # to genuinely be a ChatGPT-caliber conversational assistant for pet questions,
    # not just imitate one on a cheaper/smaller model. Path: gpt-5.4 -> gpt-5.5 ->
    # gpt-5.6-sol, each step decided by a live A/B (tone/accuracy/verbosity/memory/
    # frustration-handling scenarios, including a replay of a real customer
    # complaint about repetitive replies, plus a red-flag-emergency safety check).
    # gpt-5.6-sol won on consistent conciseness and, notably, added a sharper
    # clinical detail unprompted on the real complaint's scenario than either
    # gpt-5.5 or its sibling gpt-5.6-terra. Same reasoning_effort="none"
    # constraint applies to tool-calling turns as every gpt-5.x variant tested
    # (see chat_with_tools -- rejects function tools + a real reasoning_effort
    # together on /v1/chat/completions; only /v1/responses supports both
    # together, which this codebase doesn't use). Worth re-A/B-ing against
    # gpt-5.6-luna/terra again periodically -- these are close, and newer/less
    # proven than 5.4/5.5 were.
    openai_agent_model: str = "gpt-5.6-sol"
    openai_reasoning_model: str = "gpt-5.6-sol"
    # Voice notes / video audio tracks: gpt-4o-transcribe via the dedicated
    # /v1/audio/transcriptions endpoint, NOT gpt-audio via chat.completions
    # with input_audio (the old approach) -- confirmed live via repeated
    # real-audio testing that gpt-audio ignores the attached audio and asks
    # for it again roughly 80-90% of the time, non-deterministically, on
    # otherwise-valid clean audio. This is the direct cause of "bot doesn't
    # understand voice notes" reports. The dedicated transcription endpoint
    # was 100% reliable (multiple real clips, multiple retries each,
    # including a genuinely low-bitrate WhatsApp-realistic OGG). Trade-off:
    # transcription-only, no non-speech sound description (coughing/wheezing)
    # the old approach aimed for -- not worth keeping given how unreliable it
    # actually was; reliably hearing the words said is far more valuable.
    openai_transcription_model: str = "gpt-4o-transcribe"
    openai_agent_max_tokens: int = 5600
    openai_agent_max_iterations: int = 10
    # TTS for regional-language voice replies (see app/integrations/openai_client.py
    # synthesize_speech) -- reuses the OpenAI account already in use, no separate
    # provider/API key needed. gpt-4o-mini-tts (not tts-1) because it supports the
    # `instructions` param, needed to steer a native Indian accent instead of the
    # anglicized default -- confirmed via a user-judged 5-way A/B sample (nova was
    # the pick). Requires openai>=2.x (see requirements.txt).
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "nova"
    voice_replies_enabled: bool = True

    # Supabase
    supabase_url: str = ""
    supabase_service_role_key: str = ""

    # Google Calendar (via n8n bridge workflow — see google_calendar.py docstring)
    calendar_bridge_url: str = "https://abhilash20.app.n8n.cloud/webhook/petpulse-calendar-bridge"
    calendar_bridge_secret: str = ""

    # Razorpay (Payment Links API — a booking is only confirmed/calendared once
    # Razorpay's webhook reports the link paid, see agent/tools/booking.py)
    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    razorpay_webhook_secret: str = ""
    razorpay_consult_fee_inr: int = 399

    # Public origin this service is reachable at -- used to build shareable
    # links (e.g. GET /passport/{token}) that a vet/boarding facility can
    # open without logging in.
    public_base_url: str = "https://petpulse-python-production.up.railway.app"

    # Admin dashboard (app/admin/*) -- a single shared bearer token, checked by
    # require_admin_token. Empty means the admin API fails closed (401 on
    # everything, never silently open) until this is actually configured.
    admin_api_token: str = ""

    # Pulsy mascot welcome sticker, sent once per conversation on a bare greeting
    # (see app/agent/tools/greeting.py, GREETING_RULE). A real animated WhatsApp
    # sticker (character waving/blinking/tail-wagging, generated via Sora on a
    # flat chroma-key background then keyed to transparent webp) -- sent as a
    # `sticker` message so it loops inline in the chat like a native WhatsApp
    # sticker, not a video that opens in a player. Public Supabase Storage
    # object -- no signed URL/expiry needed, it's a static brand asset.
    pulsy_welcome_sticker_url: str = "https://ngxjkxqualvhkyyjckvs.supabase.co/storage/v1/object/public/brand-assets/pulsy-welcome.webp"

    # Doctor-onboarding-drafts sync (see app/integrations/google_drive.py,
    # app/scheduler/jobs.py sync_doctor_onboarding_drafts) -- reads a Google
    # Drive folder of per-doctor subfolders (onboarding documents: degree
    # certs, registration certificates, ID cards) via a Google Cloud service
    # account with read-only Viewer access shared on just that one folder,
    # not a personal Gmail/OAuth login. The full downloaded service-account
    # JSON key, stored as a single secret string (parsed at call time), and
    # the parent folder's Drive ID.
    google_service_account_json: str = ""
    doctor_drive_folder_id: str = ""

    # Google Places API (New) for find_nearby_vets (2026-09) -- a plain API
    # key (Maps Platform), not the service account above, since Places is
    # billed/enabled per-key on the Maps Platform project, not per service
    # account. Empty means find_nearby_vets falls back to the free OSM
    # Overpass/Nominatim path unchanged -- this is additive, not a
    # replacement that breaks when unset.
    google_maps_api_key: str = ""

    # Early-user feedback survey (2026-09-01, scripts/send_feedback_survey.py)
    # response tracking -- the Google Sheet linked to the Form's Responses
    # tab, shared Viewer with the same service account above. Empty means
    # the admin dashboard's feedback endpoint has nothing to show yet.
    feedback_survey_spreadsheet_id: str = ""

    # Per-respondent thank-you (2026-09-01, scripts/send_feedback_thankyou.py)
    # -- the feedback FORM itself is anonymous (no name/email/phone question),
    # so there is no automatic way to link a response back to a customer;
    # this sends to a specific phone number the admin has identified some
    # other way, one at a time, not a broadcast.
    whatsapp_feedback_thankyou_template_name: str = ""
    whatsapp_feedback_thankyou_template_language: str = "en"

    log_level: str = "INFO"
    timezone: str = "Asia/Kolkata"


@lru_cache
def get_settings() -> Settings:
    return Settings()
