#!/bin/bash -ex
perf record -o "$SCRATCH/perf-$SLURM_PROCID.data" "$@"
