#!/bin/bash
mkdir -p data

FILES=(
  "c4_100m.jsonl"
  "c4_validation.jsonl"
  "gpt2tok_c4_100_text_document.bin"
  "gpt2tok_c4_100_text_document.idx"
  "gpt2tok_c4_val_text_document.bin"
  "gpt2tok_c4_val_text_document.idx"
)

BASE_URL="https://huggingface.co/datasets/ZahlenReal/diffusion_data_constraint_quickstart/resolve/main"

for f in "${FILES[@]}"; do
  echo "Downloading $f ..."
  wget -O "data/${f##*/}" "${BASE_URL}/${f}?download=true"
done