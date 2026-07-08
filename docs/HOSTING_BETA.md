# Beta chat hosting

Deploy the standalone Fabrix Docs chat page (`chat/index.html`) on a Fabrix server for internal/beta feedback **before** embedding the widget on [docs.fabrix.ai](https://docs.fabrix.ai).

Same-origin nginx is recommended: one HTTPS host serves the chat UI and proxies `/health` and `/ask` to uvicorn on localhost. No CORS changes needed.

Related docs:

- Production widget embed (post-beta): [DOCS_SITE_INTEGRATION.md](DOCS_SITE_INTEGRATION.md)
- Full production checklist: [DEPLOY_CHECKLIST.md](DEPLOY_CHECKLIST.md)
- Local setup and env vars: [README.md](../README.md)

---

## Architecture

```
Browser → https://<beta-host>/
          ├── /              → static chat/index.html
          ├── /health        → proxy → uvicorn 127.0.0.1:8080
          └── /ask           → proxy → uvicorn 127.0.0.1:8080
```

The chat page defaults to `window.location.origin` for API calls when `FabrixChatConfig.apiUrl` is unset.

---

## Server prerequisites

- Python 3.9+
- Git clone of this repo
- `data/qdrant_db/` on the server (copy from a machine that ran ingest, or re-run ingest)
- `.env` with `OPENROUTER_API_KEY` and `OPENAI_API_KEY`
- TLS certificate (existing Fabrix cert or load balancer)

For same-origin beta, `DOCS_SITE_ORIGIN` can stay `*` or match the beta URL.

---

## Install

```bash
git clone <fabrix-docs-agent-repo>
cd fabrix-docs-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the project root (never commit):

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENROUTER_API_KEY` | yes | Embedding queries |
| `OPENAI_API_KEY` | yes | Answer generation |
| `DOCS_SITE_ORIGIN` | optional | CORS; `*` is fine for same-origin beta |
| `API_KEY` | optional | Require `X-API-Key` on `POST /ask` |
| `BOTS_DIR` / `DOCS_ROOT` / `CFXQL_FILE` | if re-ingesting | Override paths in `config.py` |

Pre-ingest audit (if re-ingesting):

```bash
python3 scripts/audit_ingest_sources.py
python3 src/ingest_qdrant.py
```

---

## Run the API

Bind to localhost only; nginx is the public face.

```bash
source venv/bin/activate
uvicorn src.api:app --host 127.0.0.1 --port 8080
```

Example systemd unit (`/etc/systemd/system/fabrix-docs-agent.service`):

```ini
[Unit]
Description=Fabrix Docs Agent API
After=network.target

[Service]
Type=simple
User=fabrix
WorkingDirectory=/opt/fabrix-docs-agent
EnvironmentFile=/opt/fabrix-docs-agent/.env
ExecStart=/opt/fabrix-docs-agent/venv/bin/uvicorn src.api:app --host 127.0.0.1 --port 8080
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now fabrix-docs-agent
```

**Qdrant constraint:** only one process can open `data/qdrant_db/` at a time. Do not run eval scripts while the API is up.

---

## nginx (same origin)

Example server block. Adjust paths, hostnames, and TLS paths for your environment.

```nginx
server {
    listen 443 ssl http2;
    server_name docs-agent.fabrix.ai;

    ssl_certificate     /path/to/fullchain.pem;
    ssl_certificate_key /path/to/privkey.pem;

    root /opt/fabrix-docs-agent/chat;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /health {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /ask {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
```

Reload nginx after editing:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

---

## Optional beta protection

**API key:** set `API_KEY` in `.env`, then inject before the chat script loads:

```html
<script>window.FabrixChatConfig = { apiKey: "your-secret-key" };</script>
```

**Network:** restrict the beta URL via VPN or internal DNS (no code change).

---

## Verify deploy

Replace `<beta-host>` with your HTTPS hostname (no trailing slash).

### curl

```bash
curl https://<beta-host>/health

curl -X POST https://<beta-host>/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What parameters does the count loop bot take?"}'
```

If `API_KEY` is set:

```bash
curl -X POST https://<beta-host>/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"question":"What parameters does the count loop bot take?"}'
```

### Browser

1. Open `https://<beta-host>/`
2. Status dot should show **Connected** (green)
3. Ask: *What parameters does the count loop bot take?*
4. Expect parameter list with bold names and orange `*` on required params, plus source chips linking to docs.fabrix.ai

---

## Local dev (before deploy)

Terminal 1 — API:

```bash
uvicorn src.api:app --reload --port 8080
```

Terminal 2 — chat static server:

```bash
python3 -m http.server 5173 --directory chat
```

Open `http://localhost:5173/`. The chat auto-targets `http://localhost:8080` when served from localhost on a port other than 8080.

Override explicitly:

```html
<script>window.FabrixChatConfig = { apiUrl: "http://localhost:8080" };</script>
```

---

## Beta feedback

Share with testers:

- Beta URL
- 5–10 suggested questions from `tests/eval_set.py`
- A feedback channel (Slack thread, form, or GitHub issues)

Track wrong answers, missing sources, UI confusion, latency, and questions that should refuse (e.g. subscription cancel).

**Gate before production embed:** agent eval ~19/20 pass on the hosted API and positive beta feedback.
