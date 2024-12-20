import pandas as pd
import numpy as np
import pathlib
from typing import List, Tuple, Union
from enum import Enum

class DataFileType(Enum):
    ANNOTATION = ".txt"
    HEADER = ".hea"
    AUDIO = ".wav"
    SEGMENTATION_ANNOTATION = ".tsv"
    METADATA = ".csv"

class DataColumnNotFoundException(Exception):
    """Thrown when additional columns are requested from metadata_df and they dont exist"""
    pass 

class RecordingRegistry:
    """
    Manages the metadata and audio data.
        - Keeps the metadata and organizes the labels for our problem
        - Segments the audio files into specified segment length.
            Discards files that are shorter than required length.
        - Doesn't actually load the audio files.
    """

    def __init__(self, root_dir: str):
        """

        self.data_dir: the root directory with all data (circor_dir/)
        self.metadata_df: aggregated metadata from circor_data/training_data.csv
                          then cleaned up and reorganized for murmur type classification
        """
        self.root_dir = pathlib.Path(root_dir)
        self.data_dir = self.root_dir / "training_data"
        self.metadata_path = self.root_dir / "training_data.csv"
        self.metadata_df = None

    @property
    def df(self):
        return self.metadata_df

    def load_data(self, additional_cols: List[str]=[], segment_len: float=15.0) -> pd.DataFrame:
        """
        Loads the available data in two steps and returns a dataframe where each row is one
        training example, and contains an uri column to locate the raw audio file.
        First it loads the .csv file with all annotations and assigns MurmurType labels.
        Then it explodes such that each row is one training example by expanding along the
        recording location. Multiple signals from the same recording location are identified
        by `InstanceID`. It also segments audio clips into the segments of desired length.
        Each data point turns into (audio len // `segment_len`) data points after segmenting.

        Output dataframe columns:
            - 'Patient ID': string identifier of the subject being recorded
            - 'Age': categorical string. In ("Child", "Infant", "Adolescent", "Neonate")
            - 'Sex': categorical string. In ("Male", "Female")
            - 'Height': float. In centimeters.
            - 'Weight': float. In kilograms.
            - 'Pregnancy status': Boolean. True means pregnant.
            - 'MurmurType': categorical string. In ("Systolic", "Diastolic") <=== [Label Column]
            - 'RecordingLocation': categorical string. In ("AV", "TV", "MV", "PV", "Phc")
            - 'InstanceID': int. 0 when there is a single signal per recording location (most data points)
                                counts up from 1 when there are multiple instances of same location.
                                can check if there were multiple recordings at a location by running 
                                    df[df['InstanceID'] > 0]
            - 'AudioSegment': Tuple[int]. (index of segment start, index of segment end) to clip audio.
            - 'SegmentID': int. Counts up from 0 to identify the segment number when a single audio clip
                                is split into multiple, like when audio length > k * segment_len, k > 1
            - 'Uri': str. The path to locate the raw audio file.

        Additional columns can be requested with the `additional_cols` parameter.
        """
        self.metadata_df = self._load_annotations(self.metadata_path, additional_cols=additional_cols)
        self._explode_training_examples(segment_len=segment_len)
        return self.metadata_df
        
    def _load_annotations(self, filepath: pathlib.Path, additional_cols: List[str]=[]):
        """
        Will load the metadata file, create labels for MurmurType = Systolic vs. Diastolic (& Absent)
        Then return a clean dataframe with the id columns, demographic info, and label columns
        Any additional columns required to keep can be requested with `additional_cols`.
        """
        df = pd.read_csv(filepath)
        df["Patient ID"] = df["Patient ID"].astype(str)     # is loaded as int by default

        columns_of_interest = (
            ["Patient ID"] +                                                # ids
            ["Age", "Sex", "Height", "Weight", "Pregnancy status"] +        # demographics
            ["MurmurType", "Recording locations:", "Murmur locations"]      # labels
        )
        for col in additional_cols:
            if len(col) == 0:
                continue
            if col not in df.columns:
                raise DataColumnNotFoundException(f"{col} requested but doesn't exist in metadata.")
            if col not in columns_of_interest:
                columns_of_interest.append(col)

        df["MurmurType"] = np.nan
        df.loc[df["Systolic murmur timing"].notna(), "MurmurType"] = "Systolic"
        df.loc[df["Diastolic murmur timing"].notna(), "MurmurType"] = "Diastolic"
        df = df.dropna(subset=["MurmurType"]).reset_index(drop=True)

        return df[columns_of_interest]

    def _explode_training_examples(self, segment_len: float=15.0):
        """
        Turns the records with multiple recording locations into individual data points, and updates
        their respective `MurmurType` label depending on the `Murmur locations` column.

        Then it divides data points into segment len long clips. 
        Each data point turns into (audio len // segment len) data points.
        Filters out samples less than segment len, and 

        Uses the metadata for segmenting and stores the start and end indices, 
        doesn't actually load the audio file.
        """
        self.metadata_df["RecordingLocation"] = self.metadata_df["Recording locations:"].str.split("+")        
        self.metadata_df["InstanceID"] = self.metadata_df["RecordingLocation"].apply(self._get_instance_ids_udf)
        exploded_df = self.metadata_df.explode(["RecordingLocation", "InstanceID"])
        exploded_df["MurmurType"] = exploded_df.apply(
            lambda row : (
                np.nan if pd.isna(row["Murmur locations"]) or row["RecordingLocation"] not in row["Murmur locations"] 
                else row["MurmurType"]
            ),
            axis=1
        )
        exploded_df = exploded_df.drop(["Recording locations:", "Murmur locations"], axis=1)
        exploded_df = exploded_df.dropna(subset=["MurmurType"])

        exploded_df["AudioSegment"] = exploded_df.apply(
            lambda row: self._get_segments(
                row["Patient ID"],
                row["RecordingLocation"],
                row["InstanceID"],
                segment_len
            ),
            axis = 1
        )
        exploded_df["SegmentID"] = exploded_df["AudioSegment"].apply(
            lambda segments: list(range(len(segments)))
        )
        exploded_df = exploded_df.explode(["AudioSegment", "SegmentID"])
        exploded_df["Uri"] = exploded_df.apply(
            lambda row: self._get_filename(
                DataFileType.AUDIO,
                row["Patient ID"],
                row["RecordingLocation"],
                row["InstanceID"],
                assert_exists=False
            ),
            axis=1
        )
        self.metadata_df = exploded_df.dropna(subset=["AudioSegment", "Uri"]).reset_index(drop=True)

    def _get_filename(
            self, 
            filetype: DataFileType, 
            patient_id: str, 
            recording_location: str, 
            instance_id: int, 
            assert_exists: bool=False
        ) -> str:
        """
        Constructs the data filename for the given data point.
        """
        base_name = f"{patient_id}_{recording_location}"
        if instance_id > 0:
            base_name += f"_{str(instance_id)}"
        
        filename = (self.data_dir / base_name).with_suffix(str(filetype.value))
        if not filename.exists():
            if assert_exists:
                raise FileNotFoundError(f"{filename} not found in data directory.") 
            else:
                return None    
        return filename

    #------------------------------------ UDFS -------------------------------------#
    def _get_instance_ids_udf(self, recording_locations: List[str]):
        """
        recording_locations is a list of locations like ["AV", "TV"] etc. 
        We want to number non-repeating instance ids = 0, and repeating ones with 1, 2
        """
        counts = {}
        instance_ids = []
        for loc in recording_locations:
            if recording_locations.count(loc) == 1:
                instance_ids.append(0)
            else:
                counts[loc] = counts.get(loc, 0) + 1
                instance_ids.append(counts[loc])
        return instance_ids

    def _get_segments(
            self, 
            patient_id: str, 
            recording_location: str, 
            instance_id: int, 
            segment_len: float
    ) -> List[Tuple[int]]:
        """
        Reads the .hea header file whose row1 is structures as ..... <fs> <n_samples>
        Uses the fs and n_samples to compute segment indices to satisfy segment len.
        If audio file shorter than required segment len, returns empty list.
        
        Else, returns a list of (i_start, i_end) for each cropped segment.
        
        There is no overlap between segments.        
        """
        filepath = self._get_filename(
            DataFileType.HEADER, patient_id, recording_location, instance_id, 
            assert_exists=True
        )        
        with open(filepath, "r") as f:
            row1 = f.readline().split(" ")
        fs = int(row1[-2])
        n_samples = int(row1[-1])

        n_samples_per_seg = int(fs * segment_len)
        n_segments = n_samples // n_samples_per_seg
        n_samples_available = n_samples // n_segments if n_segments > 0 else 0
        segments = []
        for i in range(n_segments):
            segment_start = i * n_samples_available + (n_samples_available - n_samples_per_seg) // 2
            segments.append((segment_start, segment_start+n_samples_per_seg))
        
        return segments
    