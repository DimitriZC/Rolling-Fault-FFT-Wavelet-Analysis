import os
import numpy as np
import scipy.io
import pywt

from scipy.signal import hilbert, butter, lfilter, firwin
from matplotlib import pyplot as plt

from typing import List, Tuple
from kurtogram import fast_kurtogram
from funcs import *

# from scipy.signal import butter, lfilter


# def band_pass(signal, low, high, fs, order=5):
#     b, a = butter(N=order, Wn=[low, high], fs=fs, btype='band')
#     return lfilter(b, a, signal)
file_to_load = 'OuterRaceFault_1.mat'
title = file_to_load.split('.')[0]

x_ORF_2, f_ORF_2, t_ORF_2, ORF_DATA = Load(file_to_load)
num = int(len(t_ORF_2)*0.0005)

freq, psd, L = PSD(x_ORF_2, t_ORF_2, num)

p_env, f_env, x_env, t_env = env_spectrum(np.transpose(x_ORF_2)[0],  f_ORF_2[0], t_ORF_2)

plot_env_spectrum(x_env, t_env, x_ORF_2, t_ORF_2, xlim=[0, 0.1])

plot_env_spectrum_analysis(f_env, p_env, ORF_DATA, BPFO=True)
plt.ylim(0, 0.2)

Kwav, Level_w, freq_w, c, max_Kurt, BW, level_max = fast_kurtogram(x_ORF_2, f_ORF_2, 9)   # Center frequency & bandwidth obtained from kurtogram
minw = np.where(Level_w == level_max)[0][0]
kurtw = np.where(Kwav[minw, :] == max_Kurt)[0][0]
bandw = freq_w[kurtw]

fc = bandw + BW[0]/2

low = fc - BW/2
high = fc + BW/2

## ERRADO: VER COMO APLICAR O FILTRO DE FREQ NESSE SINAL Q TA NO TEMPO (NO MATLAB TA ASSIM)
x_ORF_2_Bpf = band_pass(np.transpose(x_ORF_2)[0], low, high, f_ORF_2[0], order=2)

p_env_ORF_2_Bpf, f_env_ORF_2_Bpf, x_env_ORF_2_Bpf, t_env_ORF_Bpf = env_spectrum(np.transpose(x_ORF_2)[0], f_ORF_2, t_ORF_2, BAND=True, LOW=low, HIGH=high)



# plot_env_spectrum(x_env, x_ORF_2, t_ORF_2)
# plot_env_spectrum(x_env_ORF_2_Bpf, x_ORF_2_Bpf, t_ORF_2)


# plot_env_spectrum_analysis(f_env_ORF_2_Bpf, p_env_ORF_2_Bpf, ORF_DATA, BPFO=True)

high, _= hl_envelopes_idx(np.transpose(x_ORF_2)[0])
# plot_env_spectrum(x_env, t_env, x_ORF_2, t_ORF_2, HIGH=high, xlim=[0, 0.1])
# plt.ylim(-1, 1)


# high, _= hl_envelopes_idx(x_ORF_2_Bpf)
plot_env_spectrum(x_ORF_2_Bpf, t_env_ORF_Bpf, x_ORF_2, t_ORF_2, HIGH=high, xlim=[0, 0.1])
# plt.ylim(-1, 1)

plot_env_spectrum_analysis(f_env_ORF_2_Bpf, p_env_ORF_2_Bpf, ORF_DATA,BPFO=True)
plt.ylim(0, 0.2)


plt.show()

