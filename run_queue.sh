#!/bin/bash
# run all training jobs two at a time (one thread each)
cat jobs.txt | xargs -P 2 -I{} bash -c "./run_one.sh {}"
