# ==============================================================================
# Sketch2Shape AI - Makefile
# TechZephyr 4.0 Hackathon (IIT Bhubaneswar)
# ==============================================================================

.PHONY: help install install-gpu run run-mock run-share clean test

help:
	@echo "Available commands:"
	@echo "  make install        Install standard requirements"
	@echo "  make install-gpu    Install PyTorch with CUDA 12.1 and TripoSR"
	@echo "  make run            Launch the Gradio web application locally"
	@echo "  make run-mock       Launch in forced CPU Mock/Dev mode (instant testing)"
	@echo "  make run-share      Launch with public Gradio live tunnel (share=True)"
	@echo "  make clean          Clean generated meshes, caches, and temp files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-gpu:
	pip install --upgrade pip
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	pip install -r requirements.txt
	pip install git+https://github.com/VAST-AI-Research/TripoSR.git

run:
	python app.py

run-mock:
	MOCK_INFERENCE=1 python app.py

run-share:
	GRADIO_SHARE=1 python app.py

clean:
	rm -rf outputs/*.glb outputs/*.obj outputs/*.png
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
