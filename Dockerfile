FROM python:3.12-slim

# Force stdout/stderr unbuffered so logs (and crash tracebacks) show up
# immediately in PaaS log viewers instead of sitting in a pipe buffer.
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# Fixed port 8000 — this is what the currently-live Railway deploy's networking
# is already pointed at. Do not switch back to a shell-expanded ${PORT} here
# without also updating Railway's Settings > Networking target port to match,
# or the proxy and the app end up listening on two different ports (silent
# "connection refused" even though the container is healthy).
#
# --workers 4: a single worker (the previous default) means one process, one
# asyncio event loop -- and most of this app's Supabase calls are synchronous
# (blocking) network calls, not wrapped in a thread, so one customer's turn
# doing a dozen-plus DB round-trips blocks EVERY other concurrent customer's
# message on that same event loop for the duration. That's a real bottleneck
# found while investigating reports of delayed replies, and would only get
# worse at higher volume. Each worker is a separate process (its own event
# loop, its own AppContext/scheduler from the FastAPI lifespan), so this is
# real OS-level parallelism, not just concurrency within one loop. 4 is a
# starting point, not a measured optimum -- tune it against Railway's actual
# CPU/memory allocation for this service once real traffic data exists.
# Requires the scheduler jobs (app/scheduler/jobs.py) to be idempotent under
# multiple workers, which they now are (atomic per-row claim before sending) --
# don't raise this without confirming that's still true.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
