#!/bin/bash
source venv/bin/activate
export OLLAMA_GPU_ONLY=1
ollama serve &
python main.py