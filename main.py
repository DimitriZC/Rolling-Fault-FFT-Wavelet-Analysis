import numpy as np
import matplotlib.pyplot as plt


from funcs import PSD, Spec, Load

file_to_load = 'InnerRaceFault_vload_1.mat'

x_IRF_1_1, f_IRF_1_1, t_IRF_1_1 = Load(file_to_load)



num = int(len(t_IRF_1_1)*0.0005)

freq, psd, L = PSD(x_IRF_1_1, t_IRF_1_1, num)

plt.plot(freq[L], psd[L])
plt.xlim(freq[L[0]], freq[L[-1]])
plt.xlabel('Freq [Hz]')
plt.ylabel('Power Spectral Density')
plt.title('Power Spectral Density (PSD)')

Spec(np.transpose(x_IRF_1_1[:num])[0], f_IRF_1_1[0], title=file_to_load.split('.')[0])

file_to_load = 'baseline_1.mat'

x_bsln_1, f_bsln_1, t_bsln_1 = Load(file_to_load)
num = int(len(t_bsln_1)*0.0005)

Spec(np.transpose(x_bsln_1[:num])[0], f_bsln_1[0], title=file_to_load.split('.')[0])



plt.show()