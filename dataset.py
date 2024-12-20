import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from typing import List, Dict, Tuple
from data_registry import DataColumnNotFoundException
import utils

class HeartSoundDataset(Dataset):
    """
    Manages the heart sound dataset for murmur type detection. 
    Allows balanced sampling.
    Doesn't actually load the audio files.
    """
    
    def __init__(self, data_df: pd.DataFrame):
        """
        Manages the dataset described in the provided dataframe.
        Each row must be a single training example.
        
        Sanitizes the types of feature columns, it causes errors when pytorch's collate
        function combines the data points returned if we dont cause some of the features have
        nan values.
        """
        data_df["Uri"] = data_df["Uri"].astype(str)
        data_df["Age"] = data_df["Age"].astype(str)
        data_df["Sex"] = data_df["Sex"].astype(str)
        data_df["Height"] = data_df["Height"].astype(float)
        data_df["Weight"] = data_df["Weight"].astype(float)
        data_df["Pregnancy status"] = data_df["Pregnancy status"].astype(str)
        data_df["MurmurType"] = data_df["MurmurType"].astype(str)
        data_df["RecordingLocation"] = data_df["RecordingLocation"].astype(str)
        
        self.data_df = data_df
        print(f"Loaded {len(self.data_df)} training examples.")
    
    @property
    def label_dict(self) -> dict:
        """Mapping of string labels to numbers"""
        return  {
            "Diastolic": 0,
            "Systolic": 1
        }

    @property
    def inverse_label_dict(self) -> dict:
        """Mapping from numeric labels to string"""
        return {v:k for k, v in self.label_dict.items()}
    
    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, dict]:
        """
        Returns waveform and label alongside a metadata dictionary.

        Waveform is of size (N_frames,)
        Then dataloader will stack it into (N_batch, N_frames)
        """
        row = self.data_df.iloc[idx]
        waveform, _ = utils.load_audio_segment(row["Uri"], row["AudioSegment"])
        assert waveform.shape[0] == 1, "Only works with single channel audio"
        waveform = waveform.squeeze(dim=0)

        label = self.label_dict[row["MurmurType"]]
        metadata = row[
            ["Patient ID", "Age", "Sex", "Height", "Weight", "Pregnancy status", "MurmurType", "RecordingLocation"]
        ].to_dict()
        return waveform, label, metadata
    
    def get_sampler(self, balance_on: List[str], generator: torch.Generator=None) -> WeightedRandomSampler:
        """
        Creates a WeightedRandomSampler to sample without bias along the feature columns provided.
        """
        for col in balance_on:
            if col not in self.data_df.columns:
                raise DataColumnNotFoundException(f"Asked to balance on column {col}, but it doesnt exist.")
            
        joint_category = self.data_df[balance_on].astype(str).agg('_'.join, axis=1)
        frequencies = joint_category.value_counts()
        weights = 1 / frequencies[joint_category].values
        weights = weights / weights.sum()

        return WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=len(self),
            replacement=True,
            generator=generator
        )
    
def get_dataloader(
    dataset: HeartSoundDataset,
    balance_on: List[str]=["MurmurType"],
    batch_size: int = 4,
    num_workers: int = 1,
    seed: int = 42
) -> DataLoader:
    """
    create a pytorch dataLoader instance with the specified parameters
    """
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = dataset.get_sampler(balance_on, generator=generator) if balance_on is not None else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=None if sampler is not None else True,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator if sampler is None else None
    )
