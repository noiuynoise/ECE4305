from cmath import exp
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

buffer_length = 1024*32

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



ffc_input = cfc_output

# t = np.arange(0,len(data))
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

mixed_signal = []
data_out = []
for t in range(len(data)):
  
   if t == 0:
      VCO_sig_out_start = exp(1j*2*math.pi*2426e6*t/2e6)
      mixed_signal.append(ffc_input[0] * VCO_sig_out_start)
      data_out.append(ffc_input[t]*VCO_sig_out_start)
   else:
      mixed_signal.append(ffc_input[t] * VCO_sig_out_loop)
      data_out.append(ffc_input[t]*VCO_sig_out_loop)

   # plt.figure(3)
   # x_scale2 = np.linspace(-sample_rate/2, sample_rate/2, num=data.size)
   # plt.plot(x_scale2, abs(np.fft.fftshift(np.fft.fft(mixed_signal))))

   # LPF GOES HERE 

   LPF_output = signal.lfilter(num_coef, 1, mixed_signal)
   

   #take output of LPF
   theta_error = np.sign(np.cos(np.real(LPF_output[t])))*np.sin(np.real(LPF_output[t])) - np.sign(np.sin(np.real(LPF_output[t])))*np.cos(np.real(LPF_output[t]))
   # theta_error = 2*np.arccos(LPF_output[t])
   #which gives theta error for each point
   #put theta error back in as the phase for the mixer
   VCO_sig_out_loop =  exp(1j*2*math.pi*2426e6*t/2e6 + theta_error)
   #loop

   # plt.figure(5)
   # x_scale2 = np.linspace(-sample_rate/2, sample_rate/2, num=data.size)
   # plt.plot(x_scale2, abs(np.fft.fftshift(np.fft.fft(VCO_sig_out_loop))))


plt.figure(5)
x_scale2 = np.linspace(-sample_rate/2, sample_rate/2, num=data.size)
plt.plot(x_scale2, abs(np.fft.fftshift(np.fft.fft(data_out))))

plt.figure(6)
plt.plot(data_out)
plt.show()
