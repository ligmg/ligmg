#!/bin/bash -ex
#SBATCH -A m1489
#SBATCH --ntasks-per-node=4
#SBATCH --hint=memory_bound

srun "$@"
