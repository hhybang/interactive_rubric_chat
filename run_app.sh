#!/bin/bash

# Rubric Builder Web App - Mac Launcher
echo "🚀 Starting Rubric Builder Web App..."

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate rubric-builder

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

streamlit run rubric_web_app.py --server.port 8501 --server.address 0.0.0.0
