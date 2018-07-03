#!/bin/bash -ex
#SBATCH -A m1489
#SBATCH -p regular
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=END,FAIL

# ALLINEA_SAMPLER_DELAY_START=1 ALLINEA_SAMPLER_NUM_SAMPLES=5000

outfile="$1__${SLURM_JOB_NUM_NODES}_hpctoolkit"

module load hpctoolkit
srun hpcrun --event PAPI_TOT_CYC@10000 --event WALLCLOCK@100000 -o $outfile ./bin/main "${@:2}"
hpcprof -S main.hpcstruct -I src/'*' $outfile -o "$1_hpctoolkit_database"
