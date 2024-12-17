"""
Script to run the dataloader
"""
from data_loader import get_dataloader
import torch
from matplotlib import pyplot as plt
import numpy as np
import pathlib

def visualize_few_samples(dataloader: torch.utils.data.DataLoader):
    i_plot = 0
    n_plots = 3
    for i, data_point in enumerate(dataloader):
        print(f"Data point {i}...")
        f, t, Sxx = data_point
        
        print("\tpretend computing gradients")
        print("\tpretend updating model parameters")
        print("\tpretend computing validation loss")
 
        if i_plot < n_plots and i % 7 == 0:
            # plot spectogram, squeeze batch-dimension axis
            plt.clf()
            plt.pcolormesh(t.squeeze(axis=0), f.squeeze(axis=0), Sxx.squeeze(axis=0), cmap="jet")
            plt.colorbar()
            plt.xlabel("Time (sec)")
            plt.ylabel("Frequency (Hz)")
            plt.ylim(0, 1000)

            temp_dir = pathlib.Path("temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(temp_dir / f"spectogram_{i}.png")
            plt.close()
            i_plot += 1
        print("\tprocessed.")

if __name__ == '__main__':
    dataloader = get_dataloader(batch_size=1, seed=42)
    visualize_few_samples(dataloader)