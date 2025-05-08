#!/bin/bash

apptainer build --nv --writable-tmpfs --force ollama.sif  docker:ollama/ollama
my_port=$(comm -23 <(seq 9980 9999 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
echo "PORT: " $my_port
apptainer exec --nv --writable-tmpfs ollama.sif bash -c 'export OLLAMA_HOST=$(hostname -i):'$my_port' && ollama serve'
apptainer exec --nv --writable-tmps bash ollama run llama3.2

echo y | apptainer cache clean
rm ollama.sif
