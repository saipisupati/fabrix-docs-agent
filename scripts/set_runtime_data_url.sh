#!/usr/bin/env bash
# Set GitHub Actions RUNTIME_DATA_URL after you upload fabrix_runtime_data.tar.gz
# to a private HTTPS host (S3/GCS/R2 signed URL, etc.). Never commit the URL.
set -euo pipefail
URL="${1:-}"
if [[ -z "$URL" ]]; then
  echo "Usage: $0 'https://your-private-host/fabrix_runtime_data.tar.gz'" >&2
  exit 1
fi
if [[ "$URL" != https://* ]]; then
  echo "URL must start with https://" >&2
  exit 1
fi
printf '%s' "$URL" | gh secret set RUNTIME_DATA_URL
echo "Set RUNTIME_DATA_URL for $(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo 'this repo')"
gh secret list
