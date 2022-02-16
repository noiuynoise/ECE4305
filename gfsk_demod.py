import numpy as np
import math
import matplotlib.pyplot as plt
from helper_funcs import butter_lowpass_filter, butter_lowpass
from scipy import signal 


sample_rate = 2e6 # Hz
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
'''

def runpll(ffc_input, vco_gain):
    #initialize DPLL
    vco_curr_phase = 0
    vco_curr_freq = 250e3 #this is in hz - starting frequency
    #vco_gain = 10 #do math to find this

    #setup loop filter
    N = 11  # number of taps in the filter
    a = 100e3  # width of the transition band
    fc = 500e3 # cutoff frequency

    # design a halfband symmetric low-pass filter
    filter = signal.firls(N, [0, fc, fc+a, sample_rate/2], [1, 1, 0, 0], fs=sample_rate)
    loop_filter_state = signal.lfilter_zi(filter, 1)

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
        loop_filter_output, loop_filter_state = signal.lfilter(filter,1, [loop_filter_input], zi=loop_filter_state, axis=-1)
        #store loop filter output
        pll_output_array[index] = loop_filter_output[-1]
        #update VCO
        vco_curr_freq = vco_gain * loop_filter_output[-1]
    return pll_output_array

'''
for i in range(4, 8):
    output = runpll(cfc_output, 10**i)
    print(10**i)
    plt.figure()
    plt.title(f'PLL Output - gain={10**i}')
    plt.plot(timescale[6800:8000], np.angle(output[6800:8000]))
'''
pll_output_array = runpll(cfc_output, 250)

print('plotting')
fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.title('Loop Filter Output')
ax.scatter3D(timescale[22000:24000], np.real(pll_output_array[22000:24000]), np.imag(pll_output_array[22000:24000]))
plt.figure()
plt.title('PLL Output')
plt.plot(timescale[50:],np.imag(pll_output_array[50:])/np.amax(pll_output_array[50:]))
plt.figure()
plt.title('PLL Output - Phase')
plt.plot(timescale[50:],np.angle(pll_output_array[50:]))
plt.show()

