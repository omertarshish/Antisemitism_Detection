#!/bin/bash
# Script to start Ollama server on a SLURM node

# Get the node's IP address
NODE_IP=$(hostname -I | awk '{print $1}')
PORT=11434

# Start Ollama server
apptainer exec --nv /storage/containers/ollama.sif ollama serve &

# Wait for Ollama server to start
echo "Waiting for Ollama server to start..."
sleep 30

# Print server status
echo "Ollama server running at ${NODE_IP}:${PORT}"