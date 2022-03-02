import numpy as np
import adi
import matplotlib.pyplot as plt

sample_rate = 1e6 # Hz
center_freq = 2480e6 #107.9e6 # Hz

rx_bw = 1e6

buffer_length = 1024*128

data = np.fromfile('recorded_good_3.iq', np.complex64)

x_scale = np.linspace(-sample_rate/2 + center_freq, sample_rate/2 + center_freq, num=data.size)

plt.plot(x_scale, abs(np.fft.fftshift(np.fft.fft(data))))
plt.figure()

waterfall_bins = np.reshape(data, (-1, 1024))
img_data = np.zeros(waterfall_bins.shape)
for index, bin in enumerate(waterfall_bins):
    img_data[index] = np.abs(np.fft.fftshift(np.fft.fft(bin)))

plt.imshow(img_data, aspect='auto')

timescale = np.arange(0,data.size)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Sampled Data')
ax.scatter3D(timescale, np.real(data), np.imag(data))

plt.show()
