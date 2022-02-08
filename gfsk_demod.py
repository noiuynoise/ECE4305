import numpy as np
import math
import matplotlib.pyplot as plt

#load in the sampled data
data = np.fromfile('recorded.iq', np.complex64)

sample_rate = 2e6 # Hz
center_freq = 2426e6 # Hz

fft_size = 2**10
modulation_index = 2

#raised cosine / low pass filter
#TODO


cfc_input = data
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
    #find biggest bin
    maxbin = np.argmax(freq_energy)
    #add frequency offset to offsets array
    bin_offsets.append(freq_range[maxbin])

#stretch bin_offsets to size of data
bin_offsets = np.repeat(bin_offsets, fft_size)


ffc_input = cfc_trimmed
#FFC 

#initialize DPLL
dpll_curr_freq = 0
dpll_curr_phase = 0
last_sample_phase = 0

dpll_freq = [0]

freq_hi = 500e3
freq_low = -500e3

for sample in ffc_input:
    #phase determination
    sample_phase = np.angle(sample)
    #phase rotator
    dpll_curr_phase += dpll_curr_freq * 2 * np.pi / sample_rate
    rotated_phase = sample_phase + dpll_curr_phase
    #phase error detector
    sample_instant_freq = (sample_phase - last_sample_phase) * sample_rate / 2 / np.pi
    dpll_error_hf = freq_hi - sample_instant_freq
    dpll_error_lf = freq_low - sample_instant_freq
    dpll_error = dpll_error_hf
    if np.abs(dpll_error_lf) < np.abs(dpll_error_hf):
        dpll_error = dpll_error_lf
    #loop filter


#plottng
plt.plot(data / np.amax(data))

#plt.figure()

plt.plot(bin_offsets/np.amax(bin_offsets))

print(np.amax(bin_offsets))

plt.show()