import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from signal_processor import SignalProcessor
from registry import RecordingRegistry, RecordingMetadata
from typing import List, Optional, Tuple

class HeartSoundDataset(Dataset):
    """Dataset for heart sound recordings"""
    
    def __init__(self, segment_len: float=15.0):
        """
        Initialize the dataset
        
        Args:
            segment_len: Length of audio segments in seconds
        """
        print("Init dataset ...")
        self.f_lowcut = 20 # Hz
        self.f_highcut = 1000 # Hz        
        self.registry = RecordingRegistry()
        print("Registry ready...")
        self.signal_processor = SignalProcessor()
        print("Signal processor ready")
        self.recordings = self.registry.load_all_recordings(crop=True,segment_len=segment_len)
        print(f"{len(self.recordings)} recordings loaded and cropped to {segment_len} seconds")
        
    def __len__(self) -> int:
        return len(self.recordings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a processed spectrogram
        
        Returns:
            Tuple of (frequency tensor, time tensor, 2D spectrogram tensor)
        """
        recording = self.recordings[idx]
        
        # Load and normalize audio segment
        fs, audio = self.registry.load_audio_segment(recording)
        
        # Apply bandpass filter
        filtered_audio = self.signal_processor.bandpass_filter_signal(
            audio,
            fs,
            self.f_lowcut,
            self.f_highcut,
        )
        
        # Generate spectrogram
        f, t, spec_matrix = self.signal_processor.spectrogram(
            filtered_audio,
            fs,
            plot=False,
            f_max=self.f_highcut
        )
        
        # # Convert to torch tensors?
        # return (torch.from_numpy(f).float(), torch.from_numpy(t).float(), torch.from_numpy(spec_matrix).float())
        return f, t, spec_matrix

def get_dataloader(
    batch_size: int = 4,
    num_workers: int = 1,
    segment_len: float = 15.0,
    shuffle: bool = True,
    seed: int = 42
) -> DataLoader:
    """
    create a pytorch dataLoader instance with the specified parameters
    """
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    dataset = HeartSoundDataset(segment_len=segment_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator
    )