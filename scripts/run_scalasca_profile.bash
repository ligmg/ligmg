#!//bin/bash -ex
#SBATCH -A m1489
#SBATCH -p regular
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type END,FAIL
#SBATCH --hint=memory_bound

module load scalasca
SCOREP_PROFILING_MAX_CALLPATH_DEPTH=100 SCOREP_MPI_MAX_COMMUNICATORS=10000 scalasca -analyse -e "scalasca_`date +%Y-%m-%d-%H-%M-%S`" srun "$@"
