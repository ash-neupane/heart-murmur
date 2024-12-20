# Heart Murmur Detection

## Overview
A data preprocessing pipeline to train an ML model to distinguish between systolic and diastolic heart murmurs.

## Data Source
https://www.physionet.org/content/circor-heart-sound/1.0.3/ 

Audio Signals from 4 different locations in the chest, annotated.

## Environment
You may install the packages in `requirements.txt` in the virtual environment of your choice.
I was using python 3.10.16 with pip 24.2.
To initialize a conda environment:
```
conda create -n murmur_takehome
conda activate murmur_takehome
conda install python=3.10
pip install -r requirements.txt
```

## To get started
See curation_pipeline.ipynb
Don't forget to download the dataset with `download.sh`. 
If you have the data downloaded already, update the ROOT_DIR parameter in the jupyter notebook. 
That has to point to the parent directory of `training_data.csv`.


`python main.py` to run the mock training loop.

