#!/bin/bash
#SBATCH --job-name=jenova_pbt_paint_a2c
#SBATCH --account=def-foutsekh
#SBATCH --time=24:00:00
#SBATCH --mail-user=commissarsilver@gmail.com
#SBATCH --mail-type=ALL


module load NiaEnv/2019b python/3.8
mkdir /scratch/f/foutsekh/nikanjam/tprl
cd /scratch/f/foutsekh/nikanjam/tprl
virtualenv --system-site-packages /scratch/f/foutsekh/nikanjam/tprl
source /scratch/f/foutsekh/nikanjam/tprl/bin/activate

cd /scratch/f/foutsekh/nikanjam/Jenova
pip install -r requirements.txt
python tpdrl_pbt.py
