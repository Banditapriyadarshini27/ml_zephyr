#!/usr/bin/env bash
# ==============================================================================
# Sketch2Shape AI - Automated Setup Script (Linux / Google Colab / macOS)
# TechZephyr 4.0 Hackathon (IIT Bhubaneswar)
# ==============================================================================

set -e

echo "========================================================"
echo "🎨 Initializing Sketch2Shape AI Environment Setup..."
echo "========================================================"

# Create outputs directory
mkdir -p outputs
mkdir -p examples

# Optional: Initialize Python virtual environment if not in Colab
if [ -z "$COLAB_GPU" ]; then
    if [ ! -d "venv" ]; then
        echo "[1/4] Creating Python virtual environment..."
        python3 -m venv venv
    fi
    echo "[2/4] Activating virtual environment..."
    source venv/bin/activate
fi

# Upgrade pip
echo "[3/4] Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Install TripoSR from official research repo
echo "[4/4] Installing TripoSR neural reconstruction engine..."
pip install git+https://github.com/VAST-AI-Research/TripoSR.git

echo "========================================================"
echo "✅ Setup Complete!"
echo "To run locally in CPU mock mode: python app.py"
echo "To run with Gradio public share tunnel: GRADIO_SHARE=1 python app.py"
echo "========================================================"
