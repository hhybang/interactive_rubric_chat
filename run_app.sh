#!/bin/bash

# Rubric Builder Web App - Mac Launcher
echo "🚀 Starting Rubric Builder Web App..."

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate coauthor

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not found."
    echo "Please set your API key:"
    echo "export OPENAI_API_KEY=\"your-api-key-here\""
    echo ""
    read -p "Enter your OpenAI API key: " api_key
    if [ ! -z "$api_key" ]; then
        export OPENAI_API_KEY="$api_key"
    else
        echo "❌ API key is required to run the app."
        exit 1
    fi
fi

# Launch the app
echo "🌐 Launching web app at http://localhost:8501"
echo "Press Ctrl+C to stop the app"
echo ""

# Try different ports if 8501 is busy
for port in 8501 8502 8503 8504 8505; do
    if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Using port $port"
        streamlit run rubric_web_app.py --server.port $port --server.address 0.0.0.0
        break
    else
        echo "Port $port is busy, trying next..."
    fi
done
