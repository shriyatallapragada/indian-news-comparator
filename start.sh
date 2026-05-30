#!/bin/bash
# ── News Comparator — Start Script ────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
API_DIR="$PROJECT_DIR/api"
VENV="$PROJECT_DIR/.venv/bin/activate"

echo "Starting News Comparator..."

osascript <<EOF
tell application "Terminal"
    -- Tab 1: Main API (port 8000)
    do script "cd '$API_DIR' && source '$VENV' && uvicorn main:app --host 127.0.0.1 --port 8000 --reload"
    
    -- Tab 2: Engine (port 8001)
    tell application "System Events" to keystroke "t" using command down
    delay 0.5
    do script "cd '$API_DIR' && source '$VENV' && uvicorn engine:app --host 127.0.0.1 --port 8001 --reload" in front window
end tell
EOF

echo ""
echo "✅ Both servers starting:"
echo "   Port 8000 → api/main.py"
echo "   Port 8001 → api/engine.py"
