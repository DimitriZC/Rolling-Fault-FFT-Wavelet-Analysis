import os
import numpy as np
import scipy.io
import pywt

from scipy.signal import hilbert, butter, lfilter, firwin
from matplotlib import pyplot as plt

from typing import List, Tuple
from kurtogram import fast_kurtogram

from typing import List, Tuple

def Load(file: str) -> Tuple[List]:

    DATA = scipy.io.loadmat(os.path.join('Data', file))
    x = DATA['bearing']['gs'][0][0]
    fs = DATA['bearing']['sr'][0][0][0]
    time = np.transpose(np.arange(0, len(x)) / fs)

    return x, fs, time, DATA


def PSD(f: List, time: List, num: int) -> Tuple[List]:

    dt = abs(time[2] - time[1])
    n = len(time[:num])
    fhat = np.fft.fft(f[:num], n)
    freq = (1/(dt*n))*np.arange(n)
    psd = fhat*np.conj(fhat) / n
    L = np.arange(1, np.floor(n/2), dtype='int')

    return freq, psd, L

def plot_psd(freq, psd, L, title=''):

    plt.figure()
    plt.plot(freq[L], psd[L])
    plt.xlim(freq[L[0]], freq[L[-1]])
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Power Spectral Density')
    plt.title(f'PSD: {title}')




def Spec(f: List, fs: int, title: str=""):

    plt.figure()
    plt.title(title)
    plt.specgram(f, NFFT=1024, Fs=fs, noverlap=120,cmap='jet')
    plt.colorbar()


def wavelet_filter(coefficients, wavelet) -> List:
    
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coefficients)

    Csort = np.sort(np.abs(coeff_arr.reshape(-1)))

    keep = 0.05

    thresh = Csort[int(np.floor((1-keep)*len(Csort)))]
    ind = np.abs(coeff_arr) > thresh
    Cfilt = coeff_arr * ind # Threshold small indices

    coeffs_filt = pywt.array_to_coeffs(Cfilt,coeff_slices,output_format='wavedec')

    # Plot reconstruction
    Arecon = pywt.waverec(coeffs_filt,wavelet=wavelet)

    return Arecon

def band_pass(signal, low, high, fs, order=5):
    nyquist = 0.5*fs
    low = low / nyquist
    high = high / nyquist

    bpf_coefficients = firwin(5 + 1, [low[0], high[0]], pass_zero=False)
    b, a = butter(order, [low, high], fs=fs, btype='band')
    return lfilter(bpf_coefficients, 1.0,  signal)


def env_spectrum(x, fs, t, BAND=False, LOW=0, HIGH=0):
    '''
    with open("../supsub/x.csv", 'w') as f:
        f.write('time,x\n')
        for i, a in enumerate(x):
            f.write(f"{t[i]},{a}\n")
    '''
    if BAND:
        if not LOW or not HIGH:
            raise AttributeError("Frequency Band not determined. Can't be zero!")
        x = band_pass(x, LOW, HIGH, fs[0])

    analytic_signal = hilbert(np.abs(x))
    envelope = np.abs(analytic_signal)

    # high, _ = hl_envelopes_idx(np.abs(x), dmin=2)

    # envelope = np.abs(x[high])
    # t_env = t[high]
    t_env=t


    # y_d = np.gradient(np.abs(x))
    # analytic_signal = hilbert(y_d)
    # envelope = np.abs(analytic_signal)
    
    
    f_env = np.fft.fftfreq(len(envelope), d=1/fs)
    p_env = np.abs(np.fft.rfft(envelope) / len(x))
    
    
    x_env = envelope
    
    return p_env, f_env, x_env, t_env


def plot_env_spectrum(x_env, time_env, x_raw, time, HIGH=[0], xlim=[0.04, 0.06]):


    plt.figure()
    plt.subplot(2, 1, 1)
    plt.xlim(xlim)
    plt.plot(time, x_raw)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceletarion [g]')
    plt.subplot(2, 1, 2)
    if len(HIGH) > 1:
        time = time_env
    plt.plot(time, x_env)
    plt.xlim(xlim)
    # plt.ylim(0, 1)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceletarion [g]')

def plot_env_spectrum_analysis(f_env, p_env, BP, BPFI=False, BPFO=False, ALL=False):
    
    k = 0
    ncomb = 0
    if ALL: BPFI = BPFO = True

    plt.figure()
    a = next((index for index, value in enumerate(f_env) if value > 1000), -1)
    plt.plot(f_env[3:a], p_env[3:a], label='Envelope Spectrum')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel('Peak Amplitude')
    plt.xlim(0, 1000)
    plt.ylim( 0, 0.5)

    if BPFI:
        values = []
        while k * BP["BPFI"] < 1000: 
            values.append(k * BP["BPFI"])
            k +=1
        harmonics = dict(values=values, name="BPFI")
        plt.vlines(harmonics['values'], 0, 0.5, colors='r', linestyles='--', label=harmonics['name'])
    if BPFO:
        values = []
        while ncomb * BP["BPFO"] < 1000: 
            values.append(ncomb * BP["BPFO"])
            ncomb +=1
        harmonics = dict(values=values, name="BPFO")
        plt.vlines(harmonics['values'], 0, 0.5, colors='#f87915', linestyles='--', label=harmonics['name'])

    plt.legend(framealpha=1)


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Function from: https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal
    
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

