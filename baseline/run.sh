#!/bin/bash

# Test use
RUNTIME_SCRIPT_DIR="~/TencentAdv2025/baseline"
TRAIN_DATA_PATH="~/data/TencentGR_1k"

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# write your code below
python -u main.py \
  --batch_size 128 \
  --lr 0.001 \
  --maxlen 101 \
  --hidden_units 32 \
  --num_blocks 1 \
  --num_epochs 3 \
  --num_heads 1 \
  --dropout_rate 0.2 \
  --l2_emb 0.0 \
  --device 'cuda' \
  --mm_emb_id 81

# Arguments explained:
# --batch_size    Batch size (samples per batch), typical [32,512]
# --lr            Learning rate, step size for optimizer, typical [1e-4,1e-2]
# --maxlen        Max sequence length to pad/truncate sequences
# --hidden_units  Hidden layer dimension/embedding size, typical [16,128]
# --num_blocks    Number of transformer blocks (model depth), typical [1,4]
# --num_epochs    Number of epochs (full passes over data), typical [3,10]
# --num_heads     Number of attention heads in multi-head attention, [1,8]
# --dropout_rate  Dropout rate for regularization, [0.1,0.5]
# --l2_emb        L2 regularization weight for embeddings
# --device        'cuda' for GPU training or 'cpu'
# --mm_emb_id     MMemb feature IDs; choose one or more IDs from 81 to 86

# Optional flags:
# --inference_only        Only run inference without training
# --norm_first            Apply pre-layer normalization in transformer blocks
# --state_dict_path PATH  Path to pretrained model state_dict for resuming or inference