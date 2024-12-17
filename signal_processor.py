import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

class SignalProcessor:
    
    def __init__(self,):
        """
        
        """
        pass


    def bandpass_filter_signal(self, x_t: np.ndarray, fs: int, f_lowcut: int, f_highcut: int, order: int=4):
        """
        Filters signal with a butterworth bandpass filter
        """
        sos = sp.signal.butter(order, [f_lowcut, f_highcut], btype="bandpass", output = "sos", fs=fs)
        y_t = sp.signal.sosfiltfilt(sos, x_t)
        return y_t
    
    def spectrogram(self, x_t: np.ndarray, fs: int, plot: bool, f_max: int=1000):
        """
        """
        win = 0.5 # seconds
        hop = 0.8 * win # 20% overlap either side
        n_per_seg = int(fs * win)
        n_overlap = n_per_seg - int(fs * hop)
        f, t, Sxx = sp.signal.spectrogram(x_t, fs, nperseg=n_per_seg, noverlap=n_overlap, window="tukey")

        # throw away outliers and normalize
        Sxx[Sxx > np.quantile(Sxx, 0.99)] = np.quantile(Sxx, 0.99)
        for i in range(Sxx.shape[1]):
            Sxx[:,i] = (Sxx[:,i] - np.mean(Sxx[:,i])) / np.std(Sxx[:,i])

        if plot:
            plt.pcolormesh(t, f, Sxx, cmap="jet")
            plt.xlabel("Time (sec)")
            plt.ylabel("Frequency (Hz)")
            plt.ylim(0,f_max)
            plt.xlim(0,int(fs*len(x_t)))
            plt.colorbar()

        return f, t, Sxx