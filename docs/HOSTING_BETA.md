# Chat UI hosting

Deploy the standalone chat page (`chat/index.html`) on a server for testing before embedding the widget on a production documentation site.

Same-origin nginx is recommended: one HTTPS host serves the chat UI and proxies `/health` and `/ask` to uvicorn on localhost. No CORS changes needed.

Related docs:

- Widget embed: [DOCS_SITE_INTEGRATION.md](DOCS_SITE_INTEGRATION.md)
- Deploy checklist: [DEPLOY_CHECKLIST.md](DEPLOY_CHECKLIST.md)
- Local setup: [README.md](../README.md)

---

## Architecture

```
Browser → https://<your-host>/
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
- `data/kb/` on the server (copy from a machine that ran `python3 src/build_kb.py`, or rebuild after ingest)
- `.env` with `OPENROUTER_API_KEY` and `OPENAI_API_KEY`
- TLS certificate (load balancer or reverse proxy)

For same-origin hosting, `DOCS_SITE_ORIGIN` can stay `*` or match your host URL.

---

## Install

```bash
git clone <repo-url>
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
| `DOCS_SITE_ORIGIN` | optional | CORS; `*` is fine for same-origin |
| `API_KEY` | optional | Require `X-API-Key` on `POST /ask` |
| `BOTS_DIR` / `DOCS_ROOT` / `CFXQL_FILE` | if re-ingesting | Paths to public MD export |

Pre-ingest audit (if re-ingesting):

```bash
python3 scripts/audit_ingest_sources.py
python3 src/ingest_qdrant.py
python3 src/build_kb.py
```

Copy both artifacts if not rebuilding on the server:

```bash
# from build machine
tar czf fabrix_runtime_data.tar.gz data/qdrant_db data/kb
# on target host (stop API first)
tar xzf fabrix_runtime_data.tar.gz
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
User=app
WorkingDirectory=/opt/fabrix-docs-agent
EnvironmentFile=/opt/fabrix-docs-agent/.env
ExecStart=/opt/fabrix-docs-agent/venv/bin/uvicorn src.api:app --host 127.0.0.1 --port 8080
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

**Qdrant constraint:** only one process can open `data/qdrant_db/` at a time. Do not run eval scripts while the API is up.

---

## nginx (same origin)

Example server block. Adjust paths, hostnames, and TLS paths for your environment.

```nginx
server {
    listen 443 ssl http2;
    server_name docs-agent.example.com;

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

---

## Optional access control

**API key:** set `API_KEY` in `.env`, then inject before the chat script loads:

```html
<script>window.FabrixChatConfig = { apiKey: "<your-api-key>" };</script>
```

**Network:** restrict the host via firewall, allowlist, or private network as needed.

---

## Verify deploy

```bash
curl https://<your-host>/health

curl -X POST https://<your-host>/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What parameters does the count loop bot take?"}'
```

Browser: open `https://<your-host>/`, confirm Connected status, ask a bot lookup question, check sources.

---

## Local dev (before deploy)

Terminal 1 (API):

```bash
uvicorn src.api:app --reload --port 8080
```

Terminal 2 (chat static server):

```bash
python3 -m http.server 5173 --directory chat
```

Open `http://localhost:5173/`.
