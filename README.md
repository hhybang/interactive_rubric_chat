# 🍎 Rubric Builder Web App - Mac Edition

A powerful AI-powered web application for creating, managing, and comparing writing rubrics. Perfect for educators, writers, and anyone who wants to develop sophisticated rubrics through conversation with AI.

## ✨ Features

- **🤖 AI-Powered Rubric Creation**: Infer rubrics from conversations with AI writing assistants
- **📝 Visual Rubric Editor**: Easy-to-use interface for creating and editing rubrics
- **🔍 Rubric Comparison**: Side-by-side comparison of how different rubrics affect writing
- **💬 Interactive Chat**: Co-write with AI using your custom rubrics
- **📊 Prompt Logging**: Comprehensive logging of all AI interactions
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start (3 Steps)

### 1. Download & Install
```bash
# Clone or download this repository
git clone <repository-url>
cd rubric-builder-web-app

# Run the automated installer
./install_mac.sh
```

### 2. Get Your API Key
- Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
- Create a new API key
- Copy the key (you'll need it in step 3)

### 3. Launch the App
```bash
# Start the app
./run_app.sh
```

That's it! The app will be available at **http://localhost:8501**

## 📋 System Requirements

- **macOS** 10.14 or later
- **Internet connection** (for AI features)
- **OpenAI API key** (free tier available)

## 🛠️ What the Installer Does

The `install_mac.sh` script automatically:

1. **Installs Homebrew** (if not already installed)
2. **Installs Miniconda** (Python environment manager)
3. **Creates a dedicated environment** called `rubric-builder`
4. **Installs all required packages** (Streamlit, OpenAI, etc.)
5. **Sets up the data directory** structure
6. **Creates a launcher script** (`run_app.sh`)

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

## 🔧 Manual Setup (Alternative)

If you prefer to set up manually:

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3

# Add to PATH
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zprofile
source ~/.zprofile

# Create environment
conda create -n rubric-builder python=3.10 -y
conda activate rubric-builder

# Install packages
conda install -c conda-forge streamlit openai pandas matplotlib numpy tqdm -y
pip install watchdog

# Set API key
export OPENAI_API_KEY="your-api-key-here"

# Run the app
streamlit run rubric_web_app.py --server.port 8501 --server.address 0.0.0.0
```

## 🌐 Network Access

To allow others on your network to access the app:

1. **Find your IP address:**
   ```bash
   ifconfig | grep "inet "
   ```

2. **Access from other devices:**
   - Use `http://[your-ip]:8501` from any device on the same network

## 📁 File Structure

```
rubric-builder-web-app/
├── rubric_web_app.py          # Main application
├── run_app.sh                 # Launcher script
├── install_mac.sh             # Mac installer
├── requirements.txt           # Python dependencies
├── README_MAC.md             # This file
├── .gitignore                # Git ignore rules
└── rubric_demo_data2/        # Data directory
    └── .gitkeep              # Keeps directory in git
```

## 🔍 Troubleshooting

### Common Issues

**"Command not found: conda"**
```bash
# Add conda to PATH
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zprofile
source ~/.zprofile
```

**"Port 8501 already in use"**
```bash
# Use a different port
streamlit run rubric_web_app.py --server.port 8502
```

**"API key not working"**
```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set the key
export OPENAI_API_KEY="your-api-key-here"
```

**"Module not found"**
```bash
# Activate the environment
conda activate rubric-builder

# Reinstall packages
conda install -c conda-forge streamlit openai pandas matplotlib numpy tqdm -y
```

### Getting Help

1. **Check the terminal** for error messages
2. **Verify your API key** is set correctly
3. **Ensure the environment** is activated
4. **Check internet connection** for AI features

## 🔒 Security & Privacy

- **All data stored locally** on your machine
- **No data sent to external servers** except OpenAI API
- **API key stays on your machine** (not shared)
- **Network access** can be disabled by changing `0.0.0.0` to `localhost`

## 📄 License

This project is open source. Feel free to modify and distribute as needed.

## 🤝 Contributing

Found a bug or want to add a feature? Feel free to submit issues or pull requests!

---

**Ready to build better rubrics? Run `./install_mac.sh` and get started! 🚀**
