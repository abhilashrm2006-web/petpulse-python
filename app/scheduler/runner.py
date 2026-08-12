"""Wires the 3 cron jobs into an in-process APScheduler, started/stopped
from the FastAPI lifespan (spec §9 schedules: reminders/followups daily at
10:00 IST, retention weekly Sunday 3am IST)."""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.deps import AppContext
from app.scheduler.jobs import (
    flag_emergency_checkins,
    rate_customer_intents,
    retain_chat_history,
    send_doctor_schedule_reminders,
    send_new_parent_followups,
    send_onboarding_reminders,
    send_price_objection_nudges,
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
    # Same */6h cadence as reengagement_nudges (staggered 30min later),
    # since ONBOARDING_STUCK_THRESHOLD (24h) needs frequent-enough polling
    # to not let someone sit stuck for up to a day's slack past the threshold.
    scheduler.add_job(send_onboarding_reminders, CronTrigger(hour="*/6", minute=45), args=[ctx], id="onboarding_reminders")
    # No-op until google_service_account_json/doctor_drive_folder_id are set
    # (see sync_doctor_onboarding_drafts) -- safe to always register.
    scheduler.add_job(sync_doctor_onboarding_drafts, CronTrigger(hour="*/6", minute=30), args=[ctx], id="doctor_drive_sync")
    # Every 6 hours, staggered from the other */6h jobs above -- each
    # candidate here costs a real OpenAI call (see INTENT_RATING_BATCH_SIZE),
    # so this doesn't need reengagement's finer time-threshold reasoning.
    scheduler.add_job(rate_customer_intents, CronTrigger(hour="*/6", minute=50), args=[ctx], id="rate_customer_intents")
    # T-1 (evening before) and T-0 (morning of) doctor appointment-schedule
    # pushes -- two separate triggers calling the same function with a
    # different reminder_type, since "day before" and "day of" naturally
    # happen at different times of day (see send_doctor_schedule_reminders).
    scheduler.add_job(
        send_doctor_schedule_reminders, CronTrigger(hour=19, minute=0), args=[ctx, "day_before"], id="doctor_schedule_day_before"
    )
    scheduler.add_job(
        send_doctor_schedule_reminders, CronTrigger(hour=7, minute=0), args=[ctx, "day_of"], id="doctor_schedule_day_of"
    )
    # Every 6 hours -- EMERGENCY_CHECKIN_REVIEW_WINDOW (48h) is hour-granular,
    # same reasoning as reengagement_nudges above; a possible real emergency
    # shouldn't sit unflagged for up to a day's slack past the threshold.
    scheduler.add_job(flag_emergency_checkins, CronTrigger(hour="*/6", minute=5), args=[ctx], id="emergency_checkin_flags")
    # Every 6 hours, staggered from the other */6h jobs above --
    # PRICE_OBJECTION_SILENCE_THRESHOLD (6h) is itself hour-granular, same
    # reasoning as reengagement_nudges.
    scheduler.add_job(send_price_objection_nudges, CronTrigger(hour="*/6", minute=20), args=[ctx], id="price_objection_nudges")

    scheduler.start()
    return scheduler
