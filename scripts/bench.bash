#!/bin/bash
#SBATCH -A m1489
#SBATCH -p regular
#SBATCH --ntasks-per-node=4
#SBATCH --hint=memory_bound

srun ./bin/main "$@"
