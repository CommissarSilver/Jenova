#!/bin/bash
#SBATCH --job-name=tprl1
#SBATCH --account=def-foutsekh
#SBATCH --time=24:00:00
#SBATCH --mail-user=commissarsilver@gmail.com
#SBATCH --mail-type=ALL


module load NiaEnv/2019b intelpython3
conda create -n myPythonEnv python=3.6 -y
source activate myPythonEnv

cd /scratch/f/foutsekh/nikanjam/tp_rl
apt-get update && apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
pip install stable-baselines
pip install tensorflow==1.15

python testCase_prioritization/TPDRL.py -m pointwise -a A2C -t ../data/paintcontrol-additional-features.csv -e 200 -w 10 -d simple


