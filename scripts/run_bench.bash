#!/bin/bash -e

# What number of machines to benchmark on
nps="1 4 9 16 25"

# Output directory is benchmarks/graph_file_name/current_date
d=$(basename "${@: -1}")
outdir="benchmarks/${d%.*}/${NERSC_HOST}_`date +%Y-%m-%d-%H-%M-%S`"

# make sure directory exists
mkdir -p $outdir

# Write arguments to arguments.txt
echo $* > $outdir/arguments.txt

for i in $nps
do
  printf -v j "%05d" $i # format number of procs so that files are sorted in correct order
  outfile="$j.out"
  sbatch -N $i --time=00:05:00 --output="$outdir/$outfile" scripts/bench.bash --csv "$outdir/$j.csv" "$@" # force csv option
done
