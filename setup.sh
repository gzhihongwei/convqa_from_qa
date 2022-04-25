#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_md
python3 -c 'import benepar; benepar.download("benepar_en3")'
deactivate
