import numpy as np
import math
import matplotlib.pyplot as plt
from helper_funcs import butter_lowpass_filter, butter_lowpass
from scipy import signal 


sample_rate = 4e6 # Hz
center_freq = 2426e6 # Hz
fft_size = 2**10
modulation_index = 2
#load in sample data
data = np.fromfile('recorded.iq', np.complex64)
timescale = np.arange(0, data.size / sample_rate, 1/sample_rate)
#frequency shift sampled data for testing
data = data*np.exp(2j * np.pi * 250e3 * timescale)
#optional - generate fake data
#timescale = np.arange(0, 0.02, 1/sample_rate)
#data = np.exp(2j * np.pi * 250e3 * timescale) + 0.25 * np.exp(2j * np.pi * 50e3 * timescale)
print('running CFC')
#raised cosine / low pass filter
#TODO

cfc_input = data
#CFC
#trim data to nearest FFT bin
cfc_trimmed = cfc_input[0:fft_size * math.floor(cfc_input.size / fft_size)]
#put data into FFT bins
cfc_bins = np.reshape(cfc_trimmed, (-1, fft_size))
#generate FFT bin frequency array
freq_range = np.linspace(-sample_rate/2, sample_rate/2, num=fft_size)
#create recorded frequency offset array
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
bin_offsets = np.repeat(bin_offsets, fft_size)
#make a timescale for the data
freq_shift_timescale = np.arange(0, bin_offsets.size/sample_rate, 1/sample_rate)
#make frequency shifting signal to be mixed with data
freq_shift = np.exp(-1*2j*np.pi*bin_offsets*freq_shift_timescale)
#mix frequency compensation with data
cfc_output = data[0:freq_shift.size] * freq_shift
#low pass data to remove double frequency term
cfc_output = butter_lowpass_filter(cfc_output, 600e3, sample_rate)
'''
#CFC visualization
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(cfc_input))))
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
'''
print('running waterfall FFT')
fft_size = 2**4
cfc_waterfall_bins = np.reshape(cfc_output, (-1, fft_size))
cfc_data = np.zeros(cfc_waterfall_bins.shape)
for index, bin in enumerate(cfc_waterfall_bins):
    cfc_data[index] = np.abs(np.fft.fftshift(np.fft.fft(bin)))
plt.figure()
plt.title('data after CFC')
cfc_data = np.rot90(cfc_data)
plt.imshow(cfc_data,extent=[cfc_waterfall_bins.shape[1],0,-sample_rate/2, sample_rate/2], aspect='auto')


ffc_input = cfc_output
#FFC 

#initialize DPLL
vco_curr_phase = 0
vco_curr_freq = 0 #this is in hz
vco_gain = 0.1 #how do we set this?
loop_filter_cutoff = 500e3

#setup loop filter
b, a = butter_lowpass(loop_filter_cutoff, sample_rate, order=5)
loop_filter_state = signal.lfilter_zi(b, a)

print('running PLL')

#setup PLL output array
pll_output_array = np.zeros(ffc_input.shape, dtype=np.complex64)
loop_filter_input_array = np.zeros(ffc_input.shape, dtype=np.complex64)

for index,sample in enumerate(ffc_input):
    #generate VCO signal
    vco_curr_phase += vco_curr_freq * 2 * np.pi / sample_rate
    vco_output = np.exp(1j*vco_curr_phase)
    #mix VCO and input signals
    loop_filter_input = vco_output * sample
    loop_filter_input_array[index] = loop_filter_input
    loop_filter_output, zf = signal.lfilter(b,a, loop_filter_input_array[0:index+1], zi=loop_filter_state, axis=-1)
    #store loop filter output
    pll_output_array[index] = loop_filter_output[-1]
    #update VCO
    vco_curr_freq += vco_gain * loop_filter_output[-1]


print('plotting')
# NOW DECODING - Alex

# bin_one = np.exp(1j*2*np.pi*2426.185)
# bin_zero = np.exp(1j*2*np.pi*2425.815)
# samp_per_bit = 2
# for n in range(len(pll_output_array)/4):
#     k = 4*n
#     curr_bit_1 = pll_output_array[k] 
#     curr_bit_2 = pll_output_array[k-1]
#     curr_bit_3 = pll_output_array[k-2]
#     curr_bit_4 = pll_output_array[k-3]
plt.figure()
plt.title('Loop Filter Input')
plt.plot(timescale, loop_filter_input_array)

plt.figure()
plt.title('PLL Output')
plt.plot(timescale,pll_output_array/np.amax(pll_output_array))
plt.figure()
low_pass_pll_output = butter_lowpass_filter(pll_output_array, 500e3, sample_rate, order=30)
plt.title('PLL Output - Low Passed')
plt.plot(timescale,low_pass_pll_output/np.amax(low_pass_pll_output))
plt.figure()
plt.title('Input Signal')
plt.plot(ffc_input/np.amax(ffc_input))
plt.show()