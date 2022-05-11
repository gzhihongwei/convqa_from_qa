#!/bin/bash

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_md
python3 -c 'import benepar; benepar.download("benepar_en3")'

# Get SQuADv2
wget -P datasets/squad/ https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json

# Assuming that https://msropendata.com/datasets/939b1042-6402-4697-9c15-7a28de7e1321 is placed in the root of the repo, this creates the train for NewsQA
git clone https://github.com/Maluuba/newsqa.git
cp newsqa.tar.gz newsqa/maluuba/newsqa/
gdown https://drive.google.com/uc?id=0BwmD_VLjROrfTHk4NFg2SndKcjQ -O newsqa/maluuba/newsqa/
conda create --name newsqa python=2.7 "pandas>=0.19.2" -y
conda activate newsqa && pip install --requirement requirements.txt
cd newsqa
python maluuba/newsqa/data_generator.py
python -m unittest discover .
mkdir -p ../datasets/newsqa/
cp maluuba/newsqa/combined-newsqa-data-v1.json ../datasets/newsqa
cd ..
rm -rf newsqa

# Get QuAC validation
wget -P datasets/quac/ https://s3.amazonaws.com/my89public/quac/val_v0.2.json

deactivate
