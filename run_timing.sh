#!/bin/bash
# Timing-sensitive jobs, one at a time on an otherwise idle machine, with fixed glibc
# allocator thresholds (otherwise the dynamic mmap threshold adds page faults for large
# batch tensors and makes batch timings depend on the allocation history).
cd "$(dirname "$0")"
L=results/logs; mkdir -p $L
export THREADS=1 MALLOC_MMAP_THRESHOLD_=33554432 MALLOC_TRIM_THRESHOLD_=1073741824
for d in unsw cse ciciot; do python3 latency.py $d 1 > $L/latency_$d.log 2>&1; done
python3 latency.py volume cse 1 > $L/volume_cse.log 2>&1
for d in unsw cse ciciot; do python3 xai.py $d 1 > $L/xai_$d.log 2>&1; done
