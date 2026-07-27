"""FastAPI entrypoint — replaces the n8n webhook trigger + `Respond to
Webhook` verify-challenge node (spec §2). One route receives every inbound
WhatsApp event (customer or vet); everything else is delegated to
ingestion/agent modules."""

import hashlib
import hmac
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Request, Response

from app.admin.auth import require_admin_token
from app.admin.routes import router as admin_router
from app.agent.orchestrator import run_agent_turn
from app.agent.tools.booking import handle_payment_webhook
from app.agent.tools.documents import build_full_passport_text
from app.config import get_settings
from app.deps import AppContext
from app.ingestion.context import build_context
from app.ingestion.dedup import claim
from app.ingestion.media import process_media
from app.ingestion.registration import handle_registration, handle_subscription_webhook
from app.ingestion.webhook import extract_message, extract_status_update, verify_webhook_challenge
from app.integrations import razorpay_client
from app.integrations.openai_client import make_openai_client
from app.integrations.supabase_client import make_supabase_client
from app.integrations.whatsapp import WhatsAppClient
from app.scheduler.runner import start_scheduler

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        ctx = AppContext(
            settings=settings,
            http=http_client,
            whatsapp=WhatsAppClient(settings, http_client),
            supabase=make_supabase_client(settings),
            openai=make_openai_client(settings),
        )
        app.state.ctx = ctx
        scheduler = start_scheduler(ctx)
        try:
            yield
        finally:
            scheduler.shutdown(wait=False)


app = FastAPI(title="PetPulse Core Engine", lifespan=lifespan)
app.include_router(admin_router, prefix="/admin", dependencies=[Depends(require_admin_token)])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/passport/{token}")
async def public_passport(token: str, request: Request) -> Response:
    """Public, unauthenticated view of a pet's health passport -- generating
    the token is a Subscriber-only action (documents.get_shareable_link),
    but the link itself needs no login so a vet/boarding facility can open
    it directly. Renders the same text build_full_passport_text produces
    for the WhatsApp tool, so both surfaces stay in sync."""
    ctx: AppContext = request.app.state.ctx
    rows = ctx.supabase.table("pets").select("*").eq("passport_share_token", token).limit(1).execute().data
    if not rows:
        return Response(content="<h1>Link not found</h1>", media_type="text/html", status_code=404)
    passport_text, _ = build_full_passport_text(ctx.supabase, rows[0])
    html = "<html><body><pre style='font-family: sans-serif; white-space: pre-wrap;'>" + (
        passport_text.replace("*", "").replace("<", "&lt;").replace(">", "&gt;")
    ) + "</pre></body></html>"
    return Response(content=html, media_type="text/html")


@app.get("/webhook/petpulse-core")
async def verify_webhook(request: Request) -> Response:
    ctx: AppContext = request.app.state.ctx
    params = request.query_params
    challenge = verify_webhook_challenge(
        params.get("hub.mode"),
        params.get("hub.verify_token"),
        params.get("hub.challenge"),
        ctx.settings.whatsapp_verify_token,
    )
    if challenge is None:
        return Response(status_code=403)
    return Response(content=challenge, media_type="text/plain")


def _valid_signature(settings, body: bytes, signature_header: str | None) -> bool:
    if not settings.whatsapp_app_secret:
        return True  # signature verification not configured — accept (matches n8n, which didn't check it either)
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(settings.whatsapp_app_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header.removeprefix("sha256="))


@app.post("/webhook/petpulse-core")
async def receive_webhook(request: Request) -> Response:
    ctx: AppContext = request.app.state.ctx
    raw_body = await request.body()

    if not _valid_signature(ctx.settings, raw_body, request.headers.get("x-hub-signature-256")):
        return Response(status_code=403)

    body = await request.json()
    extracted = extract_message(body)
    if extracted is None or not extracted.is_valid():
        status = extract_status_update(body)
        if status is not None:
            if status.get("status") == "failed":
                logger.warning("WhatsApp delivery FAILED for message %s: %s", status.get("id"), status.get("errors"))
            else:
                logger.info("WhatsApp status update for message %s: %s", status.get("id"), status.get("status"))
        return Response(status_code=200)  # non-message webhook event (status update, etc.) — ack and ignore

    if not claim(ctx.supabase, extracted.message_id):
        return Response(status_code=200)  # duplicate delivery, already processed

    try:
        if await handle_registration(ctx, extracted):
            return Response(status_code=200)  # brand-new/mid-registration number — wizard handled it, no agent turn

        agent_ctx = await build_context(ctx.supabase, extracted)

        media_context = ""
        document_filing_status = ""
        if any([extracted.image_media_id, extracted.audio_media_id, extracted.document_media_id, extracted.video_media_id]):
            media_result = await process_media(ctx, extracted, agent_ctx.pets, agent_ctx.active_pet)
            agent_ctx.pending_media = media_result
            media_context = media_result.media_context
            if media_result.document_classification:
                document_filing_status = (
                    f"attempted; detected_type={media_result.document_classification.document_type}, "
                    f"is_medical={media_result.document_classification.is_medical_document}"
                )

        await run_agent_turn(ctx, agent_ctx, extracted, media_context, document_filing_status)
    except Exception:
        logger.exception("Failed to process inbound message %s", extracted.message_id)

    return Response(status_code=200)


@app.post("/webhook/razorpay")
async def receive_razorpay_webhook(request: Request) -> Response:
    ctx: AppContext = request.app.state.ctx
    raw_body = await request.body()

    if not razorpay_client.verify_webhook_signature(ctx.settings, raw_body, request.headers.get("x-razorpay-signature")):
        logger.warning("Rejected Razorpay webhook: signature verification failed")
        return Response(status_code=403)

    body = await request.json()
    event = body.get("event", "")
    logger.info("Received Razorpay webhook event=%s", event)
    try:
        if event.startswith("subscription."):
            handled = await handle_subscription_webhook(ctx, body)
        else:
            handled = await handle_payment_webhook(ctx, body)
        if not handled:
            logger.warning("Razorpay webhook event=%s was not acted on (see handler logs for why)", event)
    except Exception:
        logger.exception("Failed to process Razorpay webhook event")

    return Response(status_code=200)
