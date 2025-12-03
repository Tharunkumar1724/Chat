#!/bin/bash

# Setup script for Codespaces
echo "🚀 Setting up RAG Chatbot environment..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Setup frontend
echo "🎨 Setting up React frontend..."
cd frontend
npm install
cd ..

# Create data directories if they don't exist
mkdir -p data/uploads data/chunks data/index
touch data/uploads/.gitkeep data/chunks/.gitkeep data/index/.gitkeep

# Create .env template if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env template..."
    cat << EOF > .env
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# FastAPI Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Development settings
DEBUG=true
LOG_LEVEL=INFO
EOF
    echo "⚠️  Please update .env file with your Groq API key!"
fi

# Set permissions
chmod +x .devcontainer/start-services.sh

echo "✅ Setup complete! Ready to start development."
echo ""
echo "Next steps:"
echo "1. Update .env file with your Groq API key"
echo "2. Upload documents to get started"
echo "3. Run the services with: bash .devcontainer/start-services.sh"