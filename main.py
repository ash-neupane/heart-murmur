from data_registry import RecordingRegistry
from dataset import HeartSoundDataset, get_dataloader
from signal_processor import SignalProcessor
from utils import load_audio_segments
from matplotlib import pyplot as plt
import numpy as np
import pathlib
from typing import List


def main():
    # Load the data curation modules
    df = RecordingRegistry("circor_data").load_data()
    dataset = HeartSoundDataset(df)
    preprocessor = SignalProcessor()

    # Get a data loader than samples without bias on Pregnancy status
    balance_on = ["Pregnancy status",]
    dataloader = get_dataloader(
        dataset, 
        balance_on=balance_on, 
        batch_size=4, 
        seed=42
    )

    # Dictionary to count the features
    feature_counts = {}
    features_of_interest = balance_on
    if "MurmurType" not in features_of_interest:
        features_of_interest.append("MurmurType")

    # "Training" Loop
    for i, batch_data in enumerate(dataloader):
        waveform, label, metadata = batch_data
        S_tf = preprocessor.prepare_training_examples(waveform)
        print(f"Batch {i}: Spectrogram shape = {S_tf.shape}, label shape: {label.shape}")
        for feature in features_of_interest:
            if feature not in feature_counts:
                feature_counts[feature] = {}
            for value in metadata[feature]:
                feature_counts[feature][value] = feature_counts[feature].get(value, 0) + 1
        
        if i % 10 == 0:
            print(f"Feature counts after batch {i}:", feature_counts)

    print(f"Final:: Feature counts after batch {i}:", feature_counts)
    print("DONE")


def plot_spectrogram(t, f, S_tf, label, save=None):
    plt.figure(figsize=(18, 5))
    plt.pcolormesh(t, f, S_tf, cmap="viridis")
    plt.colorbar()
    plt.xlabel("Time (sec)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, 1000)
    plt.title(f"MurmurType: {label}")
    if save is not None:
        temp_dir = pathlib.Path("temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(temp_dir / save)
    plt.close()

if __name__ == '__main__':
    main()