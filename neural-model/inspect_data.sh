#!/usr/bin/env bash
JSON_PATH=$1
LINE_NUM=$2
head -n $((LINE_NUM + 1)) "$JSON_PATH" | tail -n 1 | \
    jq -r '.raw_code, ([.. | objects | select(.["node_type"] == "var") | {(.old_name): .new_name}] | unique | add)'
