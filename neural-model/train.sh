#!/usr/bin/env bash
python exp.py \
  train \
  --cuda \
  --preprocess \
  --seed=19260817 \
  --work-dir=exp_runs \
  --extra-config='{
    "data": {"train_file": "data/preprocessed_data/train-shard-*.tar" },
    "decoder": { "input_feed": false, "tie_embedding": true }, "train": { "evaluate_every_nepoch": 5, "max_epoch": 60 }
  }' \
  data/config/model.hybrid.jsonnet
