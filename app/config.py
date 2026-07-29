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

    # OpenAI
    openai_api_key: str = ""
    # Full gpt-5.4, not the -mini tier -- the "24/7 AI Health Copilot" is meant to
    # genuinely be a ChatGPT-caliber conversational assistant for pet questions, not
    # just imitate one on a cheaper/smaller model.
    openai_agent_model: str = "gpt-5.4"
    openai_reasoning_model: str = "gpt-5.4"
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

    # Razorpay Subscriptions (Subscriber membership tier, ₹399/month — see
    # app/ingestion/registration.py). Plan created once via the API; this is
    # its id, not a per-customer value.
    razorpay_subscription_plan_id: str = "plan_TIFJusQ5szdoE2"

    # Founding Member cohort (₹99/month, first FOUNDING_MEMBER_CAP sign-ups --
    # see app/agent/tools/subscriptions.py). Plan created once via the API,
    # same as razorpay_subscription_plan_id above.
    razorpay_founding_plan_id: str = "plan_TIObeBxQ0YnaEN"

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

    log_level: str = "INFO"
    timezone: str = "Asia/Kolkata"


@lru_cache
def get_settings() -> Settings:
    return Settings()
