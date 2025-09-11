#!/bin/bash

VALID_OR_TRAIN="train"
INPUT_PATH="data/TinyStoriesV2-GPT4-$VALID_OR_TRAIN.txt"
VOCAB_SIZE=10000
NUM_PROC=7
OUTPUT_DIR="tokenizer_checkpoint/tiny_$VALID_OR_TRAIN"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

uv run python cs336_basics/train_bpe.py \
   --input_path $INPUT_PATH \
   --vocab_size $VOCAB_SIZE \
   --num_processes $NUM_PROC \
   --output_dir $OUTPUT_DIR