#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=
#SBATCH --partition=short,medium
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=./out/%x_%j.out
module load singularity
singularity exec --pwd /mnt --bind ./:/mnt ./data_loader.sif python -u ./run_convert_raw_to_processed.py --data_set_name adidasvsnike