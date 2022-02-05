import numpy as np
import adi
import matplotlib.pyplot as plt
import math

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

plt.plot(data)

plt.figure()

plt.show()

t = np.arange(0,len(data))
VCO_sig_out_start = math.cos(2*math.pi*2426e6*t)
mixed_signal = data * VCO_sig_out_start

# LPF GOES HERE 


#take output if LPF, take its 2*cos^-1(LPF_output)
theta_error = 2*math.acos(LPF_output)
#which gives theta error for each point
#put theta error back in as the phase for the mixer
VCO_sig_out_loop = math.cos(2*math.pi*2426e6*t + theta_error)
#loop

