import numpy as np
import scipy as sp
import torch, torchaudio
from matplotlib import pyplot as plt
from typing import Union

class SignalProcessor:
    def __init__(
            self, 
            fs: int=4000,
            t_resolution: float=0.064,
            f_lowcut: int=20,
            f_highcut: int=1000,
            filter_order: int=4
        ):
        """
        Initializes the signal processor with the following parameters:
            - fs : Sampling frequency of the signal, default 4000 [Hz]
            - t_resolution: Time resolution of the spectrogram to generate, default 0.064 [sec]
            - f_lowcut: Lower cutoff frequency of bandpass filter, default 20 [Hz]
            - f_highcut: Higher cutoff frequency of bandpass filter, default 100 [Hz]
            - filter_order: The order of butterworth bandpass filter, default 4         
        """
        self.fs = fs

        # bandpass filter parameters
        self.f_lowcut = f_lowcut
        self.f_highcut = f_highcut
        self.filter_order = filter_order

        # spectrogram parameters
        self.t_resolution = t_resolution
        self.n_window = int(self.t_resolution * self.fs)    # default 256
        self.n_hop = int(0.5 * self.n_window)               # 50 % overlap

    def t_axis(self, n_signal: int) -> torch.Tensor:
        """
        Returns the time axis for the spectrogram in shape (N_time_slices,)
        n_signal: (int) Number of samples in the audio signal
        """
        n_slices = 1 + (n_signal - self.n_window) // self.n_hop
        return torch.arange(0, n_slices) * (self.n_hop / self.fs)

    def f_axis(self):
        """
        Returns the frequency axis for spectrogram
        """
        return torch.linspace(0, self.fs/2, self.n_window//2+1)
        
    def prepare_training_examples(self, X_t: torch.Tensor) -> torch.Tensor:
        """
        Applies a bandpass filter, then generates a spectrogram to create training examples
        from the provided batch of audio signals.

        Input:
            - X_t : signals. a 2D tensor of shape (N_batch, N_frames)

        Output:
            - S_tf: spectrograms. a 3D tensor of shape (N_batch, N_frequencies, N_time_steps)
        """
        Y_t = self.bandpass_filter_signal(X_t, tensor_out=False)
        S_tf = self.compute_spectrogram(Y_t)
        return S_tf

    def bandpass_filter_signal(self, X_t: torch.Tensor, tensor_out: bool=True) -> Union[torch.Tensor, np.ndarray]:
        """
        Filters signal with a butterworth bandpass filter using the specs
        the signal processor was initialized with. 
        Uses scipy to filter the signal so it takes data from torch tensor
        to numpy array, filters, then converts back to torch tensor.
        We use torch for data loader, audio loading etc. so, to the user of
        this function, the transformation to numpy and back is invisible.

        Reason for not using torch to filter: There are some parity issues between 
        scipy.signal.filtfilt and the implementation available in torchaudio.
        https://github.com/pytorch/audio/issues/2063
        We could definitely implement a version that works in torch if we 
        require gpu speedup for this task in the future.
        
        Input:
            - X_t : raw signals, a 2D tensor of shape (N_batch, N_frames)
            - tensor_out: [Optional] bool. the dataset/dataloader work in torch. so we want
                            the external modules to stay in torch tensors to not have to
                            move between numpy arrays and torch tensors, hence default True.
                            But, prepare_training_examples in this class immediately computes
                            spectrogram which we use scipy for. This flag provides a way to
                            avoid going to torch Tensor before immediately converting back to
                            numpy array.

        Output:
            - Y_t : bandpass filtered signals, a 2D tensor of shape (N_batch, N_frames)
        """
        sos = sp.signal.butter(
            self.filter_order, 
            [self.f_lowcut, self.f_highcut], 
            btype="bandpass", 
            output = "sos", 
            fs=self.fs
        )
        Y_t = sp.signal.sosfiltfilt(sos, X_t.numpy(), axis=-1)
        if tensor_out:
            Y_t = torch.from_numpy(Y_t.copy())

        return Y_t
        
    def compute_spectrogram(self, X_t: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Computes the spectrogram for each of the provided audio signals. 
        Using scipy to compute spectrogram, if we need 
        Input:
            - X_t : a 2D tensor of shape (N_batch, N_frames)

        Output:
            - S_tf: spectrograms. a 3D tensor of shape (N_batch, N_frequencies, N_time_steps)
        """
        if isinstance(X_t, torch.Tensor):
            X_t = X_t.numpy()

        eps = 1e-10
        _, _, S_tf = sp.signal.spectrogram(
            X_t, 
            self.fs, 
            nperseg=self.n_window, 
            noverlap=self.n_window - self.n_hop, 
            window="hamming",
            axis=-1
        )
        S_tf = np.log10(S_tf + eps)

        # throw away outliers from each batch 
        # and normalize each time slice of each
        # batch independently with z-score normalization
        quantile_99 = np.quantile(S_tf, 0.99, axis=(1,2), keepdims=True)
        S_tf = np.minimum(S_tf, quantile_99)        
        mean = np.mean(S_tf, axis=1, keepdims=True)
        std = np.std(S_tf, axis=1, keepdims=True)      
        S_tf = (S_tf - mean) / (std + eps)

        return torch.from_numpy(S_tf.copy())