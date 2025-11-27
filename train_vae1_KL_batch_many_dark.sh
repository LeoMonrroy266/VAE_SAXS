#!/bin/bash
#SBATCH --job-name=VAE_test_many_dark      # Job name
#SBATCH --output=/home/x_leomo/saxsdiff/users/Diff_SAXS_reconstruction/vae_KL_many_dark_test_log/slurm_%j.out
#SBATCH --error=/home/x_leomo/saxsdiff/users/Diff_SAXS_reconstruction/vae_KL_many_dark_test_log/slurm_%j.err
#SBATCH --gres=gpu:1                        # Number of GPUs
#SBATCH --cpus-per-task=32                   # CPU cores per task
#SBATCH --mem=64G                           # Memory
#SBATCH --time=6:00:00                     # Max runtime hh:mm:ss
#SBATCH --mail-type=END,FAIL                # Email notification on end/fail
#SBATCH --mail-user=leonardo.monrroy@kemi.uu.se  # Your email
#SBATCH --account=naiss2025-22-1083


source ~/.bashrc
conda activate diffsaxs

# Navigate to working directory
cd /home/x_leomo/saxsdiff/users/Diff_SAXS_reconstruction/debugging

# Run training
#python3 Minimal_test_VAE.py one_dark_clipped_voxels_norm/train.tfrecords
python3 VAE_late_KL.py many_dark_clipped_voxels_norm/train.tfrecords many_dark_clipped_voxels_norm/test.tfrecords $1 $2 $3 
