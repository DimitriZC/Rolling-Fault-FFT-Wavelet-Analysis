import numpy as np
import pywt
import matplotlib.pyplot as plt


from funcs import PSD, Spec, Load, wavelet_filter, env_spectrum, plot_psd

file_to_load = 'InnerRaceFault_vload_1.mat'
title = file_to_load.split('.')[0]


x_IRF_1_1, f_IRF_1_1, t_IRF_1_1, IRFDATA = Load(file_to_load)



num = int(len(t_IRF_1_1)*0.001)

freq, psd, L = PSD(x_IRF_1_1, t_IRF_1_1, num)

plot_psd(freq, psd, L, title)

Spec(np.transpose(x_IRF_1_1[:num])[0], f_IRF_1_1[0], title)

file_to_load = 'baseline_1.mat'
title = file_to_load.split('.')[0]

x_bsln_1, f_bsln_1, t_bsln_1, BSLN_DATA = Load(file_to_load)
num = int(len(t_bsln_1)*0.0005)

freq, psd, L = PSD(x_bsln_1, t_bsln_1, num)

plot_psd(freq, psd, L, title)


Spec(np.transpose(x_bsln_1[:num])[0], f_bsln_1[0], title)



#Wavelet test

wavelet='db1'
scales = (1, len(x_bsln_1))
coeffs = pywt.wavedec(x_bsln_1, wavelet)

recon = wavelet_filter(coeffs, wavelet)
title = "Recon baseline"

freq, psd, L = PSD(recon, t_bsln_1, num)

plot_psd(freq, psd, L, title)

Spec(np.transpose(recon[:num])[0], f_bsln_1[0], title='Recon')

p_env_inner, f_env_inner, x_env_inner = env_spectrum(np.transpose(x_IRF_1_1[:])[0],  f_IRF_1_1[0])

plt.figure()
plt.subplot(2, 1, 1)
plt.xlim(0.04, 0.06)
plt.plot(t_IRF_1_1, x_IRF_1_1)
plt.xlabel('Time [s]')
plt.ylabel('Acceletarion [g]')
num = int(len(t_IRF_1_1)*0.001)
plt.subplot(2, 1, 2)
plt.plot(t_IRF_1_1[:len(x_env_inner)], x_env_inner)
plt.xlim(0.04, 0.06)
# plt.ylim(0, 1)
plt.xlabel('Time [s]')
plt.ylabel('Acceletarion [g]')


plt.figure()
a = 3500
plt.plot(f_env_inner[:a], p_env_inner[:a])
plt.xlim(0, 1000)
plt.ylim( 0, 0.5)

ncomb = 10
harmonics = [k * IRFDATA["BPFI"] for k in range(1, ncomb + 1)]

plt.vlines(harmonics, 0, max(p_env_inner[:a]), colors='r', linestyles='--')


plt.show()