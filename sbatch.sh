#!/bin/sh

#SBATCH -J icn # Job name
#SBATCH -o sbatch_log/pytorch-1gpu.%j.out # Name of stdout output file (%j expands to %jobId)
#SBATCH -p A100 # queue name or partiton name titanxp/titanrtx/2080ti
#SBATCH -t 3-00:00:00 # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --gres=gpu:1 # number of gpus you want to use

#SBATCH --nodes=1
##SBATCH --exclude=n13
##SBTACH --nodelist=n12

##SBTACH --ntasks=1
##SBATCH --tasks-per-node=1
##SBATCH --cpus-per-task=1

cd $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge

echo "Start"
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export WANDB_SPAWN_METHOD=fork


nvidia-smi
date
squeue --job $SLURM_JOBID

echo "##### END #####"