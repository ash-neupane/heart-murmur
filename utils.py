import torch, torchaudio
from typing import Union, Tuple, List
import pathlib

def load_audio_segments(uris: List[str], segments: List[Tuple[int]]) -> Tuple[torch.Tensor, int]:
    """
    Loads the provided list of audio segments. 
    All audio must be single channel and must have the same sampling rate.

    Returns:
        tuple (X_t, fs)
        X_t: 2D tensor of size (N_batch, N_frames)
            where N_batch is the length of uris and segments requested.
        fs: int, sampling rate
            it requires all sampling rate to be the same
    """
    if len(uris) != len(segments):
        raise ValueError(f"Mismatching number of filenames ({len(uris)}) and segments ({len(segments)})")
    x_t_list = []
    sampling_rate = None
    for uri, segment in zip(uris, segments):
        x_t, fs = load_audio_segment(uri, segment)
        if sampling_rate is None:
            sampling_rate = fs
        else:
            if sampling_rate != fs:
                print(f"Sampling rate mismatch, ignoring: {uri} : {segment}")
                continue
        
        if x_t.shape[0] != 1:
            print(f"Only supports single channel audio, ignoring: {uri} : {segment}")
            continue

        x_t_list.append(x_t.squeeze(dim=0))
    
    return torch.stack(x_t_list, dim=0), sampling_rate

def load_audio_segment(uri: Union[str, pathlib.Path], segment: Tuple[int]) -> Tuple[torch.Tensor, int]:
    """
    Loads the audio segment specified with (index_start, index_end) in `segment` from the given `uri`
    Preferred this over scipy because it allows loading with offset and frame slicing.
    
    Returns a tuple of signal and sampling rate, 
    signal(x_t) is a 2D tensor of size N_channel x N_samples tensor
    """
    if not isinstance(uri, pathlib.Path):
        uri = pathlib.Path(uri)
    if not uri.exists():
        raise FileNotFoundError(f"{uri} not found, cant load audio.")
    
    x_t, fs = torchaudio.load(
        str(uri),
        frame_offset=segment[0], num_frames=segment[1]-segment[0], 
        normalize=True, channels_first=True
    )
    return x_t, fs
