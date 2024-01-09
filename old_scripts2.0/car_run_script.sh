#!/bin/bash
#SBATCH --job-name="CAR"
#SBATCH -p milanq #armq #milanq #fpgaq #milanq # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
##SBATCH --mem-per-cpu=1GB
#SBATCH --time=2-00:00
#SBATCH -o /home/mkkvalsu/slurm.column.%j.%N.out # STDOUT
#SBATCH -e /home/mkkvalsu/slurm.column.%j.%N.err # STDERR

module purge
module use /cm/shared/ex3-modules/latest/modulefiles
module load slurm/slurm/21.08.8

. /home/mkkvalsu/python_envs/column_env/bin/activate

srun python3 /home/mkkvalsu/projects/column/main_car_racing.py 
