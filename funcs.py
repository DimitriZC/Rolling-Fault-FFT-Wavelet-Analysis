import os
import numpy as np
import scipy.io

from scipy.signal import firwin
from matplotlib import pyplot as plt

from typing import List, Tuple
from kurtogram import fast_kurtogram

from typing import List, Tuple, Dict

def Load(file: str) -> Tuple[List]:
    """This function load information from file and parse informations

    Args:
        file (str): File Name [.mat]

    Returns:
        Tuple[List]: signal, sample rate, time, raw file data
    """

    DATA = scipy.io.loadmat(os.path.join('Data', file))
    x = DATA['bearing']['gs'][0][0]
    fs = DATA['bearing']['sr'][0][0][0]
    time = np.transpose(np.arange(0, len(x)) / fs)

    return x, fs, time, DATA


def env_spectrum(x: List[float], fs: float, time: List[float], ba: List[int, int]=0) -> Tuple[List, int]:
    """This function mocks the 'envspectrum' MATLAB function, calculating the envelope, its spectrum and frequency

    Args:
        x (List[float]): Signal
        fs (float): Sample Rate
        time (List[float]): time [s]
        ba (List[int, int], optional): Band limits to filter in demod. Defaults to 0.

    Returns:
        Tuple[List, int]: Envelope Spectrum, Frequency, Envelope Signal
    """


    envelope = complex_demod(x, fs, time, ba)

    mean_envelope = np.mean(envelope)

    envelope = envelope - mean_envelope

    
    
    f_env = np.fft.fftfreq(len(envelope), d=1/fs)
    p_env = np.abs(np.fft.fft(envelope))/len(envelope)
    
    
    x_env = envelope

    if len(p_env) % 2 != 0:
        f_env = f_env[1:int((len(f_env) + 1)/2)]
        p_env = p_env[1:int((len(p_env)+1)/2)]
        p_env[2:int(len(p_env))] = 2*p_env[2:int(len(p_env))]
    else:
        f_env = f_env[1:int(len(f_env)/2+1)]
        p_env = p_env[1:int(len(f_env)/2+1)]
        p_env[2:int(len(p_env)-1)] = 2*p_env[2:int(len(p_env)-1)]
    
    
    return p_env, f_env, x_env


def complex_demod(signal: List[float], fs: float, time: List[float], ba: List[int, int]=0, order: int=50) -> List[float]:
    """This function mocks the complex demodulation (demod) used in MATLAB to get the envelope of a signal

    Args:
        signal (List[float]): Signal on time
        fs (float): Sample Rate
        time (List[float]): time [s]
        ba (List[int, int], optional): Band limits to filter. Defaults to 0.
        order (int, optional): filter order. Defaults to 50.

    Returns:
        List[float]: Envelope Signal
    """
    

    if ba == 0:
        ba = [fs/4, 3/8*fs]
    f0 = (ba[1] + ba[0])/2
    x0 = signal*np.exp(-1j*2*np.pi*f0*time)
    b = firwin(order, (ba[1] - ba[0])/2/(fs/2))
    xAn = scipy.signal.convolve(x0, b, mode='same')
    xAn = 2*xAn
    return np.abs(xAn)



def plot_env_spectrum(x_env: List, x_raw: List, time: List, xlim: Tuple[float, float]=(0.04, 0.06)):
    """This Function plots the raw signal and its envelope

    Args:
        x_env (List): Signal Envelope
        x_raw (List): Signal on time
        time (List): time [s]
        xlim (Tuple[float, float], optional): Plot x-axis limits. Defaults to (0.04, 0.06).
    """



    plt.figure()
    plt.subplot(2, 1, 1)
    plt.xlim(xlim)
    plt.plot(time, x_raw)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceletarion [g]')
    plt.subplot(2, 1, 2)
    plt.plot(time, x_env)
    plt.xlim(xlim)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceletarion [g]')

def plot_env_spectrum_analysis(f_env: List, p_env: List, BP: Dict, BPFI: bool=False, BPFO: bool=False, ALL: bool=False):
    """This function plots the envelope Spectrum analysis, evaluating the signal amplitude in the frequency domain

    Args:
        f_env (List): Frequency domain
        p_env (List): Envelope Spectrum
        BP (Dict): File DATA
        BPFI (bool, optional): Inner Fault Harmonics. Defaults to False.
        BPFO (bool, optional): Outer Fault Harmonics. Defaults to False.
        ALL (bool, optional): Inner and Outer Fault Harmonics. Defaults to False.
    """

    
    k = 0
    ncomb = 0
    if ALL: BPFI = BPFO = True

    plt.figure()
    a = next((index for index, value in enumerate(f_env) if value > 1000), -1)
    plt.plot(f_env[3:a], p_env[3:a], label='Envelope Spectrum')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel('Peak Amplitude')
    plt.xlim(0, 1000)
    plt.ylim(0, max(p_env[3:])*1.1)


    if BPFI:
        values = []
        while k * BP["BPFI"] < 1000: 
            values.append(k * BP["BPFI"])
            k +=1
        harmonics = dict(values=values, name="BPFI")
        plt.vlines(harmonics['values'], 0, max(p_env[3:])*1.1, colors='r', linestyles='--', label=harmonics['name'])
    if BPFO:
        values = []
        while ncomb * BP["BPFO"] < 1000: 
            values.append(ncomb * BP["BPFO"])
            ncomb +=1
        harmonics = dict(values=values, name="BPFO")
        plt.vlines(harmonics['values'], 0, max(p_env[3:])*1.1, colors='#f87915', linestyles='--', label=harmonics['name'])

    plt.legend(framealpha=1)



def plot_kurtogram(x: List[float], fs: float) -> Tuple[float]:
    """This function evaluate the kurtosis spectrum and plot the kurtogram

    Args:
        x (List[float]): Signal array
        fs (float): Sample Rate

    Returns:
        Tuple[float]: center frequency and bandwidth
    """

    Kwav, Level_w, freq_w, fc, max_Kurt, bandwidth, level_max = fast_kurtogram(x, fs, 9)   # Center frequency & bandwidth obtained from kurtogram
    minw = np.where(Level_w == level_max)[0][0]
    kurtw = np.where(Kwav[minw, :] == max_Kurt)[0][0]
    bandw = freq_w[kurtw]
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(Kwav, interpolation='none', aspect='auto')
    xlavel = np.array(np.arange(0, fs/2+fs/2/7, fs/2/7), dtype=int)
    plt.xticks(np.arange(0, Kwav.shape[1], Kwav.shape[1] // 7), labels=xlavel)
    plt.title(f'K max ={max_Kurt:.4f} at level {level_max:.1f}\nCenter frequency : {bandw + bandwidth/2:.1f}Hz Bandwidth ={bandwidth:.1f}Hz')
    plt.xlabel('frequency (Hz)')
    plt.yticks(np.arange(0, Kwav.shape[0], 1), labels=np.round(Level_w, 1))
    plt.ylabel('level (window lenght)')

    plt.colorbar(im)

    plt.ylabel('Spectral Kurtosis')
    plt.tight_layout()

    return bandw, bandwidth


def complete_analysis(file: str, BPFO: bool = False, BPFI: bool = False, ALL: bool = False):
    """This function evaluate the whole analysis, reading, processing, filtering and ploting
    the results of a file

    Args:
        file (str): File to be analised
        BPFO (bool, optional): Outer Fault Harmonics. Defaults to False.
        BPFI (bool, optional): Inner Fault Harmonics. Defaults to False.
        ALL (bool, optional): Inner and Outer Fault Harmonics. Defaults to False.
    """

    title = file.split('.')[0]
    std_xlim = [0, 0.1]
    x_raw, f_raw, t_raw, DATA_raw = Load(file)

    p_env_raw, f_env_raw, x_env_raw =  env_spectrum(np.transpose(x_raw)[0], f_raw[0], t_raw)

    plot_env_spectrum((x_env_raw), x_raw, t_raw, xlim=std_xlim)
    plt.savefig(f"imgs/{title}_raw_env.png")

    plot_env_spectrum_analysis(f_env_raw, p_env_raw, DATA_raw, BPFO=BPFO, BPFI=BPFI, ALL=ALL)
    plt.title(f'Raw {title} Signal')
    plt.savefig(f"imgs/{title}_raw_envSpec.png")


    # Kurtogram analysis

    bandw, BW = plot_kurtogram(np.transpose(x_raw)[0], f_raw[0])
    plt.savefig(f"imgs/{title}_kurtogram.png")

    fc = bandw + BW/2

    low = fc - BW/2
    high = fc + BW/2

    p_env_filtered, f_env_filtered, x_env_filtered =  env_spectrum(np.transpose(x_raw)[0], f_raw[0], t_raw, ba=[low, high])

    plot_env_spectrum(x_env_filtered, x_raw, t_raw, xlim=std_xlim)
    plt.savefig(f"imgs/{title}_filt_env.png")



    plot_env_spectrum_analysis(f_env_filtered, p_env_filtered, DATA_raw, BPFO=BPFO, BPFI=BPFI, ALL=ALL)
    plt.title(f'Filtered {title} Signal')
    plt.savefig(f"imgs/{title}_filt_envSpec.png")

    plt.close()




