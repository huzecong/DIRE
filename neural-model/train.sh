#!/usr/bin/env bash
python exp.py \
  train \
  --cuda \
  --compressed \
  --seed=19260817 \
  --work-dir=exp_runs \
  --save-to=decodes \
  data/config/config.large.hybrid.jsonnet 2>&1 | tee exp_runs/log.txt
