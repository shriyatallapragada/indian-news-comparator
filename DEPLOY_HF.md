# Hugging Face Docker Space Deployment

This backend is deployed as one FastAPI app because Hugging Face Docker Spaces
only expose port `7860`.

## Files

- `Dockerfile` builds the Space container.
- `requirements.txt` installs CPU-only PyTorch on Linux and backend dependencies.
- `api/space_app.py` combines the existing `api/main.py` and `api/engine.py`
  routes into one public server.
- `.dockerignore` keeps local caches, `.env`, and Chroma files out of the image.

The Dockerfile pins `linux/amd64` because Hugging Face Spaces use that platform
and the CPU-only PyTorch wheel is not available for every local Docker
architecture, especially Apple Silicon's default Linux ARM64 build.

## Hugging Face Space Settings

Create a new Space:

- SDK: `Docker`
- Visibility: private or public
- Hardware: free CPU tier

Add these as Hugging Face Space secrets:

```text
NEWSAPI_KEY=...
GUARDIAN_API_KEY=...
GROQ_API_KEY=...
HF_TOKEN=...
```

The Docker image sets:

```text
PORT=7860
CHROMA_DB_PATH=/tmp/chroma
```

ChromaDB writes to `/tmp/chroma`, which is writable for the Space runtime user.
The contents are ephemeral on the free tier.

`HF_TOKEN` must be a Hugging Face read token from an account that has accepted
access to `ai4bharat/indic-bert`. To use a different embedding model, add this
Space variable:

```text
EMBEDDING_MODEL_NAME=some-public-or-accessible-model
```

## Extension Backend URL

In `extension/background.js`, replace:

```js
const HF_SPACE_BASE = "https://[username]-[spacename].hf.space";
```

with your Space URL, for example:

```js
const HF_SPACE_BASE = "https://yourname-indian-news-comparator.hf.space";
```

The extension will then send both existing route groups to the Space:

```js
const API_BASE = USE_HF_SPACE ? HF_SPACE_BASE : "http://127.0.0.1:8000";
const ENGINE_BASE = USE_HF_SPACE ? HF_SPACE_BASE : "http://127.0.0.1:8001";
```

Because `api/space_app.py` exposes both `/api/analyze` and
`/analyze_perspective`, no extension route names need to change.
