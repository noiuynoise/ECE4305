import numpy as np
import adi
import matplotlib.pyplot as plt

sample_rate = 2e6 # Hz
center_freq = 2426e6 #107.9e6 # Hz

rx_bw = 2e6

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.rx_lo = int(center_freq)

buffer_length = 1024*1024

sdr.rx_buffer_size = buffer_length # this is the buffer the Pluto uses to buffer samples

sdr.gain_control_mode_chan0 = "slow_attack"

data = sdr.rx()

print('got data')

data = data.astype(np.complex64)
data.tofile('recorded.iq')

x_scale = np.linspace(-sample_rate/2 + center_freq, sample_rate/2 + center_freq, num=data.size)

plt.plot(x_scale, abs(np.fft.fftshift(np.fft.fft(data))))
plt.figure()

#data_abs = np.reshape(data, (1024, -1))
#plt.imshow(np.abs(data_abs))

plt.plot(np.abs(data))

plt.figure()

plt.plot(np.angle(data))

plt.show()