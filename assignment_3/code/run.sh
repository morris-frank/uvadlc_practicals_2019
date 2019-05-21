#!/usr/bin/env bash

if [[ $(whoami) == "lgpu0248" ]]; then
    datasets="~/uvadlc_practicals_2019/assignment_3/code/data/"
else
    datasets="~/deeplearning-morris/assignment_3/code/data/"
fi

echo "#!/bin/bash

#SBATCH --job-name=test_name
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

module purge
module load eb

module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176
module load matplotlib/2.1.1-foss-2017b-Python-3.6.3
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH" > run.job

if [[ $* == *--vae* ]]; then
    echo "Running VAE with big encoding"
    echo "srun python3 vae.py --data=${datasets}" >> run.job
    rm figures/vae_*
elif [[ $* == *--manifold_vae* ]]; then
    echo "Running VAE with 2-d encoding"
    echo "srun python3 vae.py --zdim=2 --data=${datasets}" >> run.job
    rm figures/vae_*
elif [[ $* == *--gan* ]]; then
    echo "Running GAN"
    echo "srun python3 gan.py" >> run.job
fi

rm ./slurm-*
sbatch run.job