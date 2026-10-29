#!/bin/bash


# kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants
python3 -m venv Denv
. ./Denv/bin/activate


pip install seaborn
pip install scipy
pip install numpy
pip install matplotlib
pip install pandas


unzip archive.zip

python3 analysis.py

# use as a source .script.sh

