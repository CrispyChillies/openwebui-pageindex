export CORS_ALLOW_ORIGIN="http://localhost:5173;http://localhost:8080"
export WEBUI_AUTH="True"
PORT="${PORT:-8080}"
PYTHON_CMD=$(command -v python3 || command -v python)
"$PYTHON_CMD" -m uvicorn open_webui.main:app --port $PORT --host 0.0.0.0 --forwarded-allow-ips '*' --reload
