import numpy as np
import adi
import matplotlib.pyplot as plt
import matplotlib
import time

sample_rate = 54e6 # Hz
center_freq = 2.45e9 #107.9e6 # Hz

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.rx_lo = int(center_freq)
sdr.rx_buffer_size = 1024 # this is the buffer the Pluto uses to buffer samples

data = np.zeros((1024,1024))
for i in range(1024):
    data[i] = abs(np.fft.fftshift(np.fft.fft(sdr.rx())))
fg = plt.figure()
ax = fg.gca()
h = ax.imshow(data, extent=[-sample_rate/2 + center_freq, sample_rate/2 + center_freq,1024,0], aspect='auto')

roll_speed = 20

while True:
    for i in range(1024 - roll_speed):
        data[i] = data[i+roll_speed]
    for i in range(roll_speed):
        data[1024 - roll_speed + i] = abs(np.fft.fftshift(np.fft.fft(sdr.rx())))
    h.set_data(data)
    plt.draw(), plt.pause(0.01)

