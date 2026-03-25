# PageIndex Docker

Use the dedicated PageIndex container files in this fork:

```bash
docker compose -f docker-compose.pageindex.yaml up --build
```

The container runs the normal Open WebUI frontend and backend, but starts the backend with:

- `OPENWEBUI_APP_MODE=pageindex`
- PageIndex Python dependencies from `backend/pageindex/requirements.txt`
- `CHATGPT_API_KEY` mirrored from `OPENAI_API_KEY` for the legacy PageIndex utilities

Recommended environment:

```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_BASE_URL=https://api.openai.com/v1
export HF_TOKEN=your_hf_token
export PAGEINDEX_QA_MODEL=gpt-4o-mini
docker compose -f docker-compose.pageindex.yaml up --build
```

The app will be available at `http://localhost:3000`.
