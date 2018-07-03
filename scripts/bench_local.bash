#!/bin/bash -ex
# Run LAMG and ligmg on graphs in ~/laplacians and record the output
# $1 is the ligmg executable
# $2 is the suffix
# The remaining arguements are passed to ligmg

main="$1"

for f in `ls ~/laplacians/*.mtx | grep -v _R`; do
  outname=`basename "${f%.*}"`
  $main $f --csv "local_bench/$outname.ligmg.$2.csv" "${@:3}"
  # julia scripts/compare_solvers.jl --dump "local_bench/$outname.lamg.csv"
done
