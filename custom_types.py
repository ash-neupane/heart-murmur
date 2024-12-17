from dataclasses import dataclass
import pathlib
from typing import Optional, Tuple
from enum import Enum

class RecordingLocation(Enum):
    TV = "TV"
    MV = "MV"
    PV = "PV"
    AV = "AV"
    PHC = "Phc"

class MurmurStatus(Enum):
    PRESENT = 1
    ABSENT = 0
    UNKNOWN = -1

class HeartSound(Enum):
    UNANNOTATED = 0
    S1WAVE = 1
    SYSTOLIC = 2
    S2WAVE = 3
    DIASTOLIC = 4

@dataclass
class RecordingMetadata:
    """Store metadata for each recoding"""
    patient_id: str
    location: RecordingLocation
    sampling_rate: int  # Hz
    n_samples: int     # duration = sampling rate * n_samples
    # segment_window = (index_start, index_end)
    segment_window: Optional[Tuple[int]] 
    file_basename: pathlib.Path # no extension
