#! /bin/bash

python3 -m venv .venv

source .venv/bin/activate

if [ -f "requirements.txt" ];
then
    pip3 install -r requirements.txt
fi
