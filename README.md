# 🍎 Rubric Builder Web App

A powerful AI-powered web application for creating, managing, and comparing writing rubrics. Perfect for educators, writers, and anyone who wants to develop sophisticated rubrics through conversation with AI.

## ✨ Features

- **🤖 AI-Powered Rubric Creation**: Infer rubrics from conversations with AI writing assistants
- **📝 Visual Rubric Editor**: Easy-to-use interface for creating and editing rubrics
- **🔍 Rubric Comparison**: Side-by-side comparison of how different rubrics affect writing
- **💬 Interactive Chat**: Co-write with AI using your custom rubrics
- 
## 🚀 Quick Start (3 Steps)

### 1. Download & Install
```bash
# Clone or download this repository
git clone <repository-url>
cd interactive_rubric_chat

### 2. Get Your API Key
- Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
- Create a new API key
- Copy the key (you'll need it in step 3)

### 3. Create environment
conda create -n rubric-builder python=3.10 -y
conda activate rubric-builder

### 4. Install packages
conda install -c conda-forge streamlit openai pandas matplotlib numpy tqdm -y
pip install watchdog

### 5. Set API key
export OPENAI_API_KEY="your-api-key-here"

### 6. Run the app
streamlit run rubric_web_app.py --server.port {port number} --server.address 0.0.0.0
```

That's it! The app will be available at **http://localhost:{port number}**

## 🎯 How to Use

### Creating Rubrics
1. **Start a conversation** in the "Chat History" tab
2. **Chat with the AI** about your writing goals
3. **Click "Infer Rubric from Draft"** to create a rubric from the conversation
4. **Edit the rubric** using the visual editor in the sidebar

### Comparing Rubrics
1. **Go to "Compare Rubrics" tab**
2. **Select two different rubrics** from the dropdowns
3. **Enter a writing task** in the text box
4. **Click "Generate Comparison"** to see how each rubric affects the writing
5. **View the results** with visual diff highlighting

### Managing Rubrics
- **Edit criteria** using the sidebar form
- **Add/remove criteria** as needed
- **Version control** - each update creates a new version
- **Load different versions** from the dropdown
