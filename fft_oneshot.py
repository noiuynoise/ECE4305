from cmath import exp
import numpy as np
import adi
import matplotlib.pyplot as plt
import math
from scipy import signal 

sample_rate = 1e6 # Hz
center_freq = 2426e6 

rx_bw = 1e6

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
data.tofile('recorded_1MHz_3.iq')

x_scale = np.linspace(-sample_rate/2 + center_freq, sample_rate/2 + center_freq, num=data.size)
plt.figure(1)
plt.plot(x_scale, abs(np.fft.fftshift(np.fft.fft(data))))

plt.figure(2)
plt.plot(data)

fft_size = 2**10
modulation_index = 2

cfc_input = data
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(cfc_input))))
#CFC
cfc_trimmed = cfc_input[0:fft_size * math.floor(cfc_input.size / fft_size)]
cfc_bins = np.reshape(cfc_trimmed, (-1, fft_size))

freq_range = np.linspace(-sample_rate/2, sample_rate/2, num=fft_size)
bin_offsets = []

for bin in cfc_bins:
    #remove modulation components
    bin_no_mod = np.power(bin, modulation_index) #i think there is a better algorithm here
    #take FFT
    freq_energy = np.abs(np.fft.fftshift(np.fft.fft(bin_no_mod)))
    #find total of all bins
    all_energy = np.sum(freq_energy)
    #find 50% bin
    curr_energy = 0
    curr_bin = 0
    while curr_energy < all_energy / 2:
        curr_energy += freq_energy[curr_bin]
        curr_bin += 1
    #add frequency offset to offsets array
    bin_offsets.append(freq_range[curr_bin])

#stretch bin_offsets to size of data
bin_offsets = np.repeat(bin_offsets, fft_size) * -1

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

freq_shift_timescale = np.arange(0, bin_offsets.size/sample_rate, 1/sample_rate)
freq_shift = np.exp(2j*np.pi*bin_offsets*freq_shift_timescale)


cfc_output = data[0:freq_shift.size] * freq_shift


cfc_output = butter_lowpass_filter(cfc_output, 500e3, 2e6)

cfc_waterfall_bins = np.reshape(cfc_output, (-1, fft_size))
cfc_data = np.zeros(cfc_waterfall_bins.shape)
for index, bin in enumerate(cfc_waterfall_bins):
    cfc_data[index] = np.abs(np.fft.fftshift(np.fft.fft(bin)))

pre_cfc_data = np.zeros(cfc_bins.shape)
for index, bin in enumerate(cfc_bins):
    pre_cfc_data[index] = np.abs(np.fft.fftshift(np.fft.fft(bin)))
    
plt.figure(3)
plt.title('data after CFC')
plt.imshow(cfc_data,extent=[-sample_rate/2, sample_rate/2,cfc_waterfall_bins.shape[1],0], aspect='auto')
plt.figure(4)
plt.title('data before CFC')
plt.imshow(pre_cfc_data,extent=[-sample_rate/2, sample_rate/2,cfc_waterfall_bins.shape[1],0], aspect='auto')




ffc_input = cfc_output
plt.show()
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

