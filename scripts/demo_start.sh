#!/usr/bin/env bash
# demo_start.sh — proxy-safe local demo boot (API :8080 + chat :5173)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Broken corporate proxies cause OpenAI Connection refused mid-demo
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" || -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY and OPENROUTER_API_KEY must be set (e.g. in .env)"
  exit 1
fi

if [[ ! -d venv ]]; then
  echo "ERROR: venv/ missing — create with: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

# Admin + chat both need this when set. Default for local demo only.
if [[ -z "${API_KEY:-}" ]]; then
  API_KEY=local-dev
  echo "API_KEY unset — using local-dev for this session"
fi

mkdir -p logs
pkill -f 'uvicorn src.api:app' 2>/dev/null || true
pkill -f 'http.server 5173 --directory chat' 2>/dev/null || true
sleep 1
rm -f data/qdrant_db/.lock 2>/dev/null || true

nohup env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  OPENAI_API_KEY="$OPENAI_API_KEY" \
  OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  API_KEY="$API_KEY" \
  uvicorn src.api:app --host 127.0.0.1 --port 8080 \
  >> logs/uvicorn-demo.log 2>&1 &
echo $! > logs/uvicorn-demo.pid

nohup python3 -m http.server 5173 --directory chat --bind 127.0.0.1 \
  >> logs/chat-demo.log 2>&1 &
echo $! > logs/chat-demo.pid

echo "Waiting for /health..."
for _ in $(seq 1 30); do
  if curl -sf -m 2 http://127.0.0.1:8080/health | grep -q ok; then
    echo "API OK  → http://127.0.0.1:8080/health"
    echo "Chat UI → http://127.0.0.1:5173/"
    echo "Admin   → http://127.0.0.1:5173/admin.html  (API_KEY=$API_KEY)"
    echo "Save that API_KEY once on the admin page so chat /ask works."
    echo "Logs: logs/uvicorn-demo.log  logs/chat-demo.log"
    echo "Stop: kill \$(cat logs/uvicorn-demo.pid) \$(cat logs/chat-demo.pid)"
    exit 0
  fi
  sleep 1
done

echo "ERROR: API did not become healthy — see logs/uvicorn-demo.log"
tail -40 logs/uvicorn-demo.log || true
exit 1
