#!/bin/bash

# Rubric Builder Web App - Mac Installation Script
# This script sets up everything needed to run the app on macOS

set -e  # Exit on any error

echo "🍎 Rubric Builder Web App - Mac Installation"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS only."
    exit 1
fi

print_status "Checking system requirements..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    print_warning "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon Macs
    if [[ $(uname -m) == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    print_success "Homebrew found"
fi

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_warning "Conda not found. Installing Miniconda..."
    
    # Download and install Miniconda
    if [[ $(uname -m) == "arm64" ]]; then
        # Apple Silicon
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
        bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3
        rm Miniconda3-latest-MacOSX-arm64.sh
    else
        # Intel
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
        bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda3
        rm Miniconda3-latest-MacOSX-x86_64.sh
    fi
    
    # Add conda to PATH
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zprofile
    source ~/.zprofile
else
    print_success "Conda found"
fi

# Initialize conda
print_status "Initializing conda..."
eval "$(conda shell.bash hook)"

# Create conda environment
print_status "Creating conda environment 'rubric-builder'..."
if conda env list | grep -q "rubric-builder"; then
    print_warning "Environment 'rubric-builder' already exists. Removing it..."
    conda env remove -n rubric-builder -y
fi

conda create -n rubric-builder python=3.10 -y

# Activate environment
print_status "Activating environment..."
conda activate rubric-builder

# Install required packages
print_status "Installing required packages..."
conda install -c conda-forge streamlit openai pandas matplotlib numpy tqdm -y

# Install additional packages via pip
print_status "Installing additional packages..."
pip install watchdog

# Create data directory
print_status "Setting up data directory..."
mkdir -p rubric_demo_data2
touch rubric_demo_data2/.gitkeep

# Set up API key
print_status "Setting up OpenAI API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    print_warning "No OpenAI API key found in environment variables."
    echo "You'll need to set your API key to use the app."
    echo ""
    read -p "Enter your OpenAI API key now (or press Enter to skip): " api_key
    if [ ! -z "$api_key" ]; then
        echo "export OPENAI_API_KEY=\"$api_key\"" >> ~/.zprofile
        export OPENAI_API_KEY="$api_key"
        print_success "API key saved to ~/.zprofile"
    else
        print_warning "API key not set. You'll need to set it before running the app."
    fi
else
    print_success "OpenAI API key found in environment"
fi

# Create launcher script
print_status "Creating launcher script..."
cat > run_app.sh << 'EOF'
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
EOF

chmod +x run_app.sh

# Create requirements.txt
print_status "Creating requirements.txt..."
conda list -e > requirements.txt

# Final instructions
echo ""
print_success "Installation complete! 🎉"
echo ""
echo "To run the app:"
echo "1. Open Terminal"
echo "2. Navigate to this directory"
echo "3. Run: ./run_app.sh"
echo ""
echo "Or manually:"
echo "1. conda activate rubric-builder"
echo "2. streamlit run rubric_web_app.py --server.port 8501 --server.address 0.0.0.0"
echo ""
echo "The app will be available at: http://localhost:8501"
echo ""

# Test if everything works
print_status "Testing installation..."
if python -c "import streamlit, openai, pandas" 2>/dev/null; then
    print_success "All packages installed correctly!"
else
    print_error "Some packages failed to install. Please check the output above."
    exit 1
fi

print_success "Ready to go! Run ./run_app.sh to start the app."
