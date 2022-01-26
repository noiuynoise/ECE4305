import numpy as np
import adi
import matplotlib.pyplot as plt

sample_rate = 54e6 # Hz
center_freq = 2445.5e6 #107.9e6 # Hz

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.rx_lo = int(center_freq)

buffer_length = 1024*4

sdr.rx_buffer_size = buffer_length # this is the buffer the Pluto uses to buffer samples

num_buffers = 500

sdr.gain_control_mode_chan0 = "slow_attack"
data = np.zeros((num_buffers,buffer_length))
for i in range(num_buffers):
    data[i] = abs(np.fft.fftshift(np.fft.fft(sdr.rx())))
fg = plt.figure()
ax = plt.axes([0.10, 0.42, 0.8, 0.55])
print_data = data / np.amax(data)
h = ax.imshow(print_data, extent=[-sample_rate/2 + center_freq, sample_rate/2 + center_freq,1024,0], aspect='auto', vmin=0, vmax=1)
ax_spectrogram = plt.axes([0.10, 0.12, 0.8, 0.2])
x_scale = np.linspace(-sample_rate/2 + center_freq, sample_rate/2 + center_freq, num=data[num_buffers - 1].size)
spect, = ax_spectrogram.plot(x_scale,data[num_buffers - 1])
ax_slider = plt.axes([0.20, 0.01, 0.65, 0.03])
slider = plt.Slider(ax_slider, 'Slide->', 70, 6000, valinit = center_freq / 1e6)

roll_speed = 10

def update_center(val):
    print(val)
    center_freq = val * 1000000
    sdr.rx_lo = int(center_freq)
    h.set(extent = [-sample_rate/2 + center_freq, sample_rate/2 + center_freq,1024,0])

slider.on_changed(update_center)

brightness = np.amax(data)

while True:
    for i in range(num_buffers - roll_speed):
        data[i] = data[i+roll_speed]
    for i in range(roll_speed):
        data[num_buffers - roll_speed + i] = abs(np.fft.fftshift(np.fft.fft(sdr.rx())))
    print_data = data / np.amax(data)
    h.set_data(print_data)
    x_scale = np.linspace(-sample_rate/2 + center_freq, sample_rate/2 + center_freq, num=data[num_buffers - 1].size)
    spect.set_data(x_scale, data[num_buffers - 1])
    plt.draw(), plt.pause(0.01)

