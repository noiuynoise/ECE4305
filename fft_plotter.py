import numpy as np
import math
import matplotlib.pyplot as plt
from helper_funcs import butter_lowpass_filter, butter_lowpass
from scipy import signal 


data = np.fromfile('recorded.iq', np.complex64)

plt.title('input FFT')
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(data))))
plt.show()