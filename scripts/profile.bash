#!/bin/bash -ex
#SBATCH -A m1489
#SBATCH -p regular
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type END,FAIL
#SBATCH --hint=memory_bound

# ALLINEA_SAMPLER_DELAY_START=1 ALLINEA_SAMPLER_NUM_SAMPLES=5000

outfile="$1__$SLURM_JOB_NUM_NODES.map"

module load allineatools
map -o $outfile --profile --procs-per-node=4 -n $(($SLURM_JOB_NUM_NODES*4)) bin/main-profile "${@:2}"
