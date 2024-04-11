#!/bin/bash
#SBATCH --job-name=rz95_inference # Job name
#SBATCH -o ./slurm_output.out -e ./slurm_error.err
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=10
#SBATCH --mem=1G                     # Job memory request
#SBATCH --time=1:00:00               # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:2
hostname && nvidia-smi && env
python3 ./diffusion_example.py 

