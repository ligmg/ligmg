#!//bin/bash -ex
#SBATCH -A m1489
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type END,FAIL
#SBATCH --hint=memory_bound

srun ./scripts/perf.bash "$@"
