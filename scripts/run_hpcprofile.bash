#!/bin/bash -e

d=$(basename "${@: -1}")
outdir="profiling/${d%.*}"
mkdir -p $outdir
prefix="$outdir/${NERSC_HOST}_`date +%Y-%m-%d-%H-%M-%S`"
echo $prefix

sbatch -N $1 --output="$prefix.out" scripts/hpcprofile.bash $prefix "${@:2}"
