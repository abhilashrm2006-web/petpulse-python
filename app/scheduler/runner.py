"""Wires the 3 cron jobs into an in-process APScheduler, started/stopped
from the FastAPI lifespan (spec §9 schedules: reminders/followups daily at
10:00 IST, retention weekly Sunday 3am IST)."""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.deps import AppContext
from app.scheduler.jobs import (
    retain_chat_history,
    send_new_parent_followups,
    send_reengagement_nudges,
    send_vaccination_reminders,
    sync_doctor_onboarding_drafts,
)


def start_scheduler(ctx: AppContext) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=ctx.settings.timezone)

    scheduler.add_job(send_vaccination_reminders, CronTrigger(hour=10, minute=0), args=[ctx], id="vaccination_reminders")
    scheduler.add_job(send_new_parent_followups, CronTrigger(hour=10, minute=0), args=[ctx], id="new_parent_followups")
    scheduler.add_job(retain_chat_history, CronTrigger(day_of_week="sun", hour=3, minute=0), args=[ctx], id="chat_history_retention")
    # Every 6 hours rather than once a day -- the 48h silence threshold is
    # itself hour-granular, so a once-daily check could let a customer sit
    # silent for up to ~72h (48h + up to a day's slack) before being noticed.
    scheduler.add_job(send_reengagement_nudges, CronTrigger(hour="*/6", minute=15), args=[ctx], id="reengagement_nudges")
    # No-op until google_service_account_json/doctor_drive_folder_id are set
    # (see sync_doctor_onboarding_drafts) -- safe to always register.
    scheduler.add_job(sync_doctor_onboarding_drafts, CronTrigger(hour="*/6", minute=30), args=[ctx], id="doctor_drive_sync")

    scheduler.start()
    return scheduler
