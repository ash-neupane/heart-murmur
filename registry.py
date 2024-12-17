import pandas as pd
import numpy as np
import pathlib
import scipy as sp
import math 
from metadata import RecordingLocation, RecordingMetadata
from typing import Tuple, List

class RecordingTooShortException(Exception):
    """We plan to fix the segment length of training examples. Thrown if the audio is not long enough"""
    pass

class RecordingRegistry:
    """
    Manages the metadata and audio data.
        - Keeps the metadata
        - Loads the audio files as requested into numpy arrays
        - Segments the audio files into specified segment length.
            Discards files that are shorter than required length.
        - TODO: when segmenting, make sure to include the annotated part
                currently, just segments the middle 15 seconds

    """

    def __init__(self,):
        """

        self.data_dir: the root directory with all data (circor_dir/)
        self.metadata_df: aggregated metadata from circor_data/training_data.csv
        self.records: The base filename of all records: <path_to_file>/<patient_id>_<rec_loc>, 
                        add .hea, .wav, .tsv to get specific files.
                        replace _<rec_loc> with .txt extension for the annotation file.
        """
        self.root_dir = pathlib.Path("circor_data")
        self.data_dir = self.root_dir / "training_data"
        self.metadata_df = pd.read_csv("circor_data/training_data.csv")
        
        records_file = self.root_dir / "RECORDS"
        if not records_file.exists():
            raise FileNotFoundError(f"Records file {records_file} not found.")
        
        with open("circor_data/RECORDS", "r") as f:
            records = f.readlines()
        self.records = [self.root_dir / r.strip() for r in records]
        print(f"{len(self.records)} records available.")

    def load_all_recordings(self, crop: bool=False, segment_len: float=None) -> List[RecordingMetadata]:
        """
        Loads all available audio recordings. 
        Doesn't actually load the audio files which would be long arrays in memory, loads the
        metadata including a crop window to load the actual audio file later on-demand using `load_audio_segment`
        """            
        recordings = []
        n_samples_skipped = 0
        for rec_basename in self.records:
            rec_header = rec_basename.with_suffix(".hea")
            if not rec_header.exists():
                print(f"\t Header {rec_header} not found.")
                continue
            with open(rec_header, "r") as f:
                row1 = f.readline().split(" ")

            fs = int(row1[-2])
            n_samples = int(row1[-1])

            if crop:
                if segment_len is None:
                    raise ValueError("segment_len must be provided when crop is True.")

                n_samples_needed = math.ceil(segment_len * fs)
                if n_samples < n_samples_needed:
                    n_samples_skipped += 1
                    continue

                segment_start = (n_samples - n_samples_needed) // 2
            else:
                n_samples_needed = n_samples
                segment_start = 0

            patient_id = rec_basename.name.split("_")[0]
            location = RecordingLocation(rec_basename.name.split("_")[1])
            recordings.append(RecordingMetadata(
                patient_id = patient_id,
                location = location,
                sampling_rate = fs,
                n_samples = n_samples,
                segment_window = (segment_start, segment_start+n_samples_needed),
                file_basename=rec_basename
            ))
        print(f"\tCropped {len(recordings)} recordings, {n_samples_skipped} skipped.")
        return recordings

    def load_audio_segment(self, rec_metadata: RecordingMetadata):
        """
        Loads the audio segment associated with rec_metadata
        """
        wav_filename = rec_metadata.file_basename.with_suffix(".wav")
        assert wav_filename.exists(), f"Audio file {wav_filename} not found."
        fs, x_t = sp.io.wavfile.read(wav_filename)
        normalizer = 32768.0 # 16 bit samples norm to -1 to 1. =2^(16-1)

        assert x_t.dtype == np.int16, f"Expecting to work with 16 bit sampling, got {x_t.dtype}"
        assert rec_metadata.n_samples == len(x_t), f"Mismatch in audio len {len(x_t)} and header info {rec_metadata.n_samples}"
        assert rec_metadata.sampling_rate == fs, f"Mismatch in sampling rate {fs} and header info {rec_metadata.sampling_rate}"

        i_start, i_end = rec_metadata.segment_window
        return fs, x_t[i_start:i_end]/normalizer