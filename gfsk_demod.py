import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal 

#load in the sampled data
data = np.fromfile('recorded.iq', np.complex64)



sample_rate = 2e6 # Hz
center_freq = 2426e6 # Hz

fft_size = 2**10
modulation_index = 2

timescale = np.arange(0, data.size / sample_rate, 1/sample_rate)

data = data*np.exp(2j * np.pi * 250e3 * timescale)

#timescale = np.arange(0, 0.02, 1/sample_rate)
#data = np.exp(2j * np.pi * 250e3 * timescale) + 0.25 * np.exp(2j * np.pi * 50e3 * timescale)

#raised cosine / low pass filter
#TODO


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
    
plt.figure()
plt.title('data after CFC')
plt.imshow(cfc_data,extent=[-sample_rate/2, sample_rate/2,cfc_waterfall_bins.shape[1],0], aspect='auto')
plt.figure()
plt.title('data before CFC')
plt.imshow(pre_cfc_data,extent=[-sample_rate/2, sample_rate/2,cfc_waterfall_bins.shape[1],0], aspect='auto')
plt.show()


ffc_input = cfc_output
#FFC 

#initialize DPLL
dpll_curr_freq = 0
dpll_curr_phase = 0
last_sample_phase = 0

loop_filter_integrator = 0

dpll_kp = 1000 #hz per rad offset
dpll_ki = 10

dpll_freq = [0]

freq_hi = 500e3
freq_low = -500e3

for sample in ffc_input:
    #phase determination
    sample_phase = np.angle(sample)
    #phase rotator
    dpll_curr_phase += dpll_curr_freq * 2 * np.pi / sample_rate
    rotated_phase = sample_phase + dpll_curr_phase
    phase_delta = sample_phase - last_sample_phase
    if phase_delta > np.pi:
        phase_delta = 2*np.pi - phase_delta
    if phase_delta < -1*np.pi:
        phase_delta = -2*np.pi - phase_delta
    #phase error detector
    sample_instant_freq = phase_delta * sample_rate / 2 / np.pi
    dpll_error_hf = freq_hi - sample_instant_freq
    dpll_error_lf = freq_low - sample_instant_freq
    dpll_error = dpll_error_hf
    if np.abs(dpll_error_lf) < np.abs(dpll_error_hf):
        dpll_error = dpll_error_lf

    #loop filter
    #just proportional for now
    loop_filter_integrator += dpll_error * dpll_ki
    dpll_curr_freq = loop_filter_integrator + dpll_error * dpll_kp

    #variable setting
    dpll_freq.append(dpll_curr_freq)
    last_sample_phase = sample_phase

'''
plt.figure()

plt.plot(ffc_input)

plt.figure()

plt.plot(np.array(dpll_freq))

plt.show()
'''