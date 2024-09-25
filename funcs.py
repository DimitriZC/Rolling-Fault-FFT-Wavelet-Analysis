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

    bpf_coefficients = firwin(50 + 1, [low, high], pass_zero=False)
    b, a = butter(order, [low, high], fs=fs, btype='band')
    return lfilter(bpf_coefficients, 1.0, signal)


def env_spectrum(x, fs, time, ba=0):


    # analytic_signal = hilbert(x)
    # envelope = np.abs(analytic_signal)

    envelope = complex_demod(x, fs, time, ba)

    mean_envelope = np.mean(envelope)



    # Remove DC bias by subtracting the mean from the envelope
    envelope = envelope - mean_envelope

    # high, _ = hl_envelopes_idx(np.abs(x), dmin=2)

    # envelope = np.abs(x[high])
    # t_env = t[high]


    # y_d = np.gradient(np.abs(x))
    # analytic_signal = hilbert(y_d)
    # envelope = np.abs(analytic_signal)
    
    
    f_env = np.fft.fftfreq(len(envelope), d=1/fs)
    p_env = np.abs(np.fft.fft(envelope))/len(envelope)
    
    
    x_env = envelope

        # Compute one-sided spectrum. Compensate the amplitude for a two-sided
    # spectrum. Double all points except DC and nyquist.
    if len(p_env) % 2 != 0:
        # Odd length two-sided spectrum
        f_env = f_env[1:int((len(f_env) + 1)/2)]
        p_env = p_env[1:int((len(p_env)+1)/2)]
        p_env[2:int(len(p_env))] = 2*p_env[2:int(len(p_env))]
    else:
        # Even length two-sided spectrum
        f_env = f_env[1:int(len(f_env)/2+1)]
        p_env = p_env[1:int(len(f_env)/2+1)]
        p_env[2:int(len(p_env)-1)] = 2*p_env[2:int(len(p_env)-1)]
    
    
    return p_env, f_env, x_env


def complex_demod(signal, fs, time, ba=0, order=50):

    if ba == 0:
        ba = [fs/4, 3/8*fs]
    f0 = (ba[1] + ba[0])/2
    x0 = signal*np.exp(-1j*2*np.pi*f0*time)
    b = firwin(order, (ba[1] - ba[0])/2/(fs/2))
    xAn = scipy.signal.convolve(x0, b, mode='same')
    xAn = 2*xAn
    return np.abs(xAn)



def plot_env_spectrum(x_env, x_raw, time, xlim=[0.04, 0.06]):


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


def plot_kurtogram(x, fs) -> Tuple[float]:

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

    x_filtered = band_pass(np.transpose(x_raw)[0], low, high, f_raw[0])

    p_env_filtered, f_env_filtered, x_env_filtered =  env_spectrum(np.transpose(x_raw)[0], f_raw[0], t_raw, ba=[low, high])

    plot_env_spectrum(x_env_filtered, x_raw, t_raw, xlim=std_xlim)
    plt.savefig(f"imgs/{title}_filt_env.png")



    plot_env_spectrum_analysis(f_env_filtered, p_env_filtered, DATA_raw, BPFO=BPFO, BPFI=BPFI, ALL=ALL)
    plt.title(f'Filtered {title} Signal')
    plt.savefig(f"imgs/{title}_filt_envSpec.png")

    plt.close()




