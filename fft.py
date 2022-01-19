import numpy as np
import adi
import matplotlib.pyplot as plt
import matplotlib
import time

sample_rate = 54e6 # Hz
center_freq = 425e6 #107.9e6 # Hz

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.rx_lo = int(center_freq)
sdr.rx_buffer_size = 1024 # this is the buffer the Pluto uses to buffer samples

data = np.zeros((1024,1024))

for i in range(1024):
    data[i] = abs(np.fft.fft(sdr.rx()))
    #time.sleep(0.01)
freq = np.fft.fftfreq(1024, d=1/sample_rate)
print(freq)
plt.imshow(data)
plt.show()
print(data)

