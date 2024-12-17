# Heart Murmur Detection

## Overview
A data preprocessing pipeline to train an ML model to distinguish between systolic and diastolic heart murmurs.

## Data Source
https://www.physionet.org/content/circor-heart-sound/1.0.3/ 

Audio Signals from 4 different locations in the chest, annotated.

## Preprocessing
Crop the audio signals to 15 seconds and apply a 20Hz to 1000Hz bandpass filter (butterworth).

## Modeling

## Mitigating Demographic Bias
Age is imbalanced globally. Sex seems balanced globally but when we filter for available labels, it could be imbalanced.

## Scaling Up
