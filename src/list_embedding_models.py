"""
list_embedding_models.py, dev utility: print OpenRouter embedding models list.
"""

import os
import requests

response = requests.get(
    "https://openrouter.ai/api/v1/embeddings/models",
    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
)

if response.status_code == 200:
    models = response.json().get("data", [])
    print(f"Found {len(models)} embedding models:\n")
    for m in models:
        pricing = m.get("pricing", {})
        print(f"  {m['id']:<45} pricing={pricing}")
else:
    print(f"Error {response.status_code}: {response.text}")

