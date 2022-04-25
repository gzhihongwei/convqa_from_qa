#!/bin/bash

#SBATCH -j NP-squad
#SBATCH -o output/squad/%j.log
#SBATCH -e output/squad/%j.err
#SBATCH -p {partition}
#SBATCH -t {time}
#SBATCH -c {cpus}
#SBATCH --mem {memory}

DATASETS_DIR=$(readlink -f $0 | xargs dirname | xargs dirname)/datasets

source venv/bin/activate
python3 preprocess_squad.py $DATASETS_DIR/squad/{file}
deactivate
