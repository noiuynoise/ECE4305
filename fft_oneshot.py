import numpy as np
import adi
import matplotlib.pyplot as plt
import math
from scipy import signal 

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
plt.figure(1)
plt.plot(x_scale, abs(np.fft.fftshift(np.fft.fft(data))))

plt.figure(2)
plt.plot(data)

t = np.arange(0,len(data))
VCO_sig_out_start = np.cos(2*math.pi*2426e6*t)
mixed_signal = data * VCO_sig_out_start

plt.figure(3)
x_scale2 = np.linspace(-sample_rate/2, sample_rate/2, num=data.size)
plt.plot(x_scale2, abs(np.fft.fftshift(np.fft.fft(mixed_signal))))

# LPF GOES HERE 
num_coef = [-0.006991626021578, -0.01754311971686, -0.01272215069694, 0.006912181652354,
   0.009615955044781,-0.009791312677446,-0.009167351758746,  0.01465913219682,
   0.008545472360693, -0.02156225944864, -0.00624978403779,  0.03080483620668,
   0.001051555498208,  -0.0438086622689, 0.009896246546655,  0.06428226310897,
   -0.03448288429489,  -0.1092798106379,   0.1227298709107,   0.4738650475032,
     0.4738650475032,   0.1227298709107,  -0.1092798106379, -0.03448288429489,
    0.06428226310897, 0.009896246546655,  -0.0438086622689, 0.001051555498208,
    0.03080483620668, -0.00624978403779, -0.02156225944864, 0.008545472360693,
    0.01465913219682,-0.009167351758746,-0.009791312677446, 0.009615955044781,
   0.006912181652354, -0.01272215069694, -0.01754311971686,-0.006991626021578]
LPF_output = signal.lfilter(num_coef, 1, mixed_signal)
plt.figure(4)
x_scale2 = np.linspace(-sample_rate/2, sample_rate/2, num=data.size)
plt.plot(x_scale2, abs(np.fft.fftshift(np.fft.fft(LPF_output))))

#take output if LPF, take its 2*cos^-1(LPF_output)
theta_error = 2*np.arccos(LPF_output)
#which gives theta error for each point
#put theta error back in as the phase for the mixer
VCO_sig_out_loop = np.cos(2*math.pi*2426e6*t + theta_error)
#loop

plt.figure(5)
x_scale2 = np.linspace(-sample_rate/2, sample_rate/2, num=data.size)
plt.plot(x_scale2, abs(np.fft.fftshift(np.fft.fft(VCO_sig_out_loop))))

plt.figure(6)
plt.plot(data)

plt.show()
