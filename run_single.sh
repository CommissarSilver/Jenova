!/bin/bash
#SBATCH --job-name=mpi_job

module load NiaEnv/2019b python/3.8
mkdir ~/.virtualenvs
cd ~/.virtualenvs
virtualenv --system-site-packages ~/.virtualenvs/tprl
source ~/.virtualenvs/tprl/bin/activate

git clone https://github.com/CommissarSilver/Jenova
cd Jenova
pip install -r requirements.txt
python tpdrl.py
