from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from biosignal_toolbox.utils import getAbsolutePath


raw_data = np.load(getAbsolutePath("src/JTE_Project/offline/filter_tests/raw_online.npy"))
bp_data = np.load(getAbsolutePath("src/JTE_Project/offline/filter_tests/bandpass_online.npy"))
raw_offline_data = np.load(getAbsolutePath("src/JTE_Project/offline/filter_tests/raw_offline.npy"))
bp_offline_data = np.load(getAbsolutePath("src/JTE_Project/offline/filter_tests/bandpass_offline.npy"))

C = raw_data.shape[0]
N = raw_data.shape[1]
T = 1.0 / 500.0

print(N)

t = np.linspace(0, N*T, N)
f = np.linspace(0, 1.0/(2*T), N//2)

raw_f = fft(raw_data)
bp_f = fft(bp_data)
raw_off_f = fft(raw_offline_data)
bp_off_f = fft(bp_offline_data)

H_bp = np.divide(bp_f, raw_f)
H_off_bp = np.divide(bp_off_f, raw_off_f)

for i in range(C):
    fig, ax = plt.subplots(4)
    fig.suptitle(f'Channel {i+1}')
    ax[0].plot(f, (2.0/N)*np.abs(raw_f[i,:N//2]), "tab:green")
    #ax[0].plot(f, (2.0/N)*np.abs(raw_off_f[i,:N//2]), "tab:blue")
    ax[0].set_title("Raw-FFT")
    ax[1].plot(f, (2.0/N)*np.abs(bp_f[i,:N//2]), "tab:green")
    #ax[1].plot(f, (2.0/N)*np.abs(bp_off_f[i,:N//2]), "tab:blue")
    ax[1].set_title("BP-FFT")
    ax[2].plot(f, np.abs(H_bp[i,:N//2]), "tab:orange")
    #ax[2].plot(f, np.abs(H_off_bp[i,:N//2]), "tab:red")
    ax[2].set_title("|H_BP|")
    ax[3].plot(f, (180/np.pi)*np.angle(H_bp[i,:N//2]), "tab:red")
    #ax[3].plot(f, (180/np.pi)*np.angle(H_off_bp[i,:N//2]), "tab:blue")
    ax[3].set_title("arg(H_BP)")

    plt.show()


