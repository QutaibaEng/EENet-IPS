#!/bin/bash
# usage: run_one.sh python3 <script> <args...>; skips jobs whose result file exists
script=$2; shift 2
if [ "$script" = "train.py" ]; then out="results/$1_$2_s$3.json"; else out="results/$1_fl_$2_s$3.json"; fi
tag=$(echo "$script $@" | tr ' ' '_' | sed 's/\.py//')
[ -f "$out" ] && exit 0
mkdir -p results/logs
THREADS=1 python3 $script "$@" > results/logs/$tag.log 2>&1
