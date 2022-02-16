import numpy as np
import adi
import matplotlib.pyplot as plt

sample_rate = 2e6 # Hz
center_freq = 2480e6 #107.9e6 # Hz

rx_bw = 1e6

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(rx_bw) # filter cutoff, just set it to the same as sample rate
sdr.rx_lo = int(center_freq)

buffer_length = 1024*32

sdr.rx_buffer_size = buffer_length # this is the buffer the Pluto uses to buffer samples

sdr.gain_control_mode_chan0 = "slow_attack"

data = sdr.rx()

print('got data')

data = data.astype(np.complex64)
data.tofile('recorded.iq')

x_scale = np.linspace(-sample_rate/2 + center_freq, sample_rate/2 + center_freq, num=data.size)

plt.plot(x_scale, abs(np.fft.fftshift(np.fft.fft(data))))
plt.figure()

waterfall_bins = np.reshape(data, (-1, 1024))
img_data = np.zeros(waterfall_bins.shape)
for index, bin in enumerate(waterfall_bins):
    img_data[index] = np.abs(np.fft.fftshift(np.fft.fft(bin)))

plt.imshow(img_data, aspect='auto')

plt.show()
