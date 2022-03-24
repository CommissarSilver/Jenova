#!/bin/bash
#SBATCH --job-name=mpi_job
#SBATCH --account=def-foutsekh
#SBATCH --time=24:00:00
#SBATCH --mail-user=commissarsilver@gmail.com
#SBATCH --mail-type=ALL


module load NiaEnv/2019b python/3.8
mkdir /scratch/f/foutsekh/nikanjam
cd /scratch/f/foutsekh/nikanjam
virtualenv --system-site-packages /scratch/f/foutsekh/nikanjam/tprl
source /scratch/f/foutsekh/nikanjam/tprl/bin/activate

git clone https://github.com/CommissarSilver/Jenova
cd Jenova
pip install -r requirements.txt
python tpdrl.py
