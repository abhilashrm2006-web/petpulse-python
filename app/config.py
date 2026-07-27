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
    openai_audio_model: str = "gpt-audio"
    openai_agent_max_tokens: int = 5600
    openai_agent_max_iterations: int = 10

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
    # see app/agent/tools/subscriptions.py). Placeholder until the plan is
    # actually created via the API (blocked as of this writing by an expired
    # Razorpay key -- see razorpay_key_id/secret) -- replace once real.
    razorpay_founding_plan_id: str = "plan_FOUNDING_PLACEHOLDER"

    # Public origin this service is reachable at -- used to build shareable
    # links (e.g. GET /passport/{token}) that a vet/boarding facility can
    # open without logging in.
    public_base_url: str = "https://petpulse-python-production.up.railway.app"

    log_level: str = "INFO"
    timezone: str = "Asia/Kolkata"


@lru_cache
def get_settings() -> Settings:
    return Settings()
