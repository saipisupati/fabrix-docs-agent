# Docs site integration

Embed the Fabrix docs Q&A widget on a documentation site.

## Paste this into your HTML

```html
<link rel="stylesheet" href="/path/to/ask-widget.css">
<script src="/path/to/ask-widget.js" data-api-url="https://your-api-host.example.com"></script>
```

Optional: mount inside a specific element instead of `body`:

```html
<div id="docs-ask"></div>
<script src="/path/to/ask-widget.js" data-api-url="https://your-api-host.example.com" data-container="docs-ask"></script>
```

Local dev (API on port 8080):

```html
<link rel="stylesheet" href="/widget/ask-widget.css">
<script src="/widget/ask-widget.js" data-api-url="http://localhost:8080"></script>
```

## API URL

Point `data-api-url` (or `window.FabrixAskConfig.apiUrl`) at the hosted FastAPI server base URL (no trailing slash). The widget calls `POST {apiUrl}/ask`.

Production must use **HTTPS**.

## CORS

On the API server, set:

```bash
DOCS_SITE_ORIGIN=https://your-docs-site.example.com
```

This allows browser requests from the docs site. Local dev defaults to `*` when unset.

## API key (optional)

On the server:

```bash
API_KEY=<generate-a-secret>
```

In the page, before loading the widget script:

```html
<script>
  window.FabrixAskConfig = {
    apiUrl: "https://your-api-host.example.com",
    apiKey: "<same-secret-as-API_KEY>"
  };
</script>
<script src="/path/to/ask-widget.js"></script>
```

The widget sends `X-API-Key` on each `/ask` request. If `API_KEY` is unset on the server, no header is required.

## What stays server-side

Never put these in the widget or public HTML:

- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`
- `API_KEY` (unless loaded via server-side templating you control; prefer env on API only)
- Qdrant path / `data/qdrant_db/`

The widget only talks to your FastAPI `/ask` endpoint.
