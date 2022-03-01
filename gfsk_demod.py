import numpy as np
import math
import matplotlib.pyplot as plt
from helper_funcs import butter_lowpass_filter, butter_lowpass
from scipy import signal 


sample_rate = 2e6 # Hz
center_freq = 2426e6 # Hz
fft_size = 2**14
modulation_index = 2
#load in sample data
data = np.fromfile('recorded.iq', np.complex64)
timescale = np.arange(0, data.size / sample_rate, 1/sample_rate)
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
freq_shift = np.exp(-1j*np.pi*bin_offsets*freq_shift_timescale)
#mix frequency compensation with data
#cfc_output = data[0:freq_shift.size] * freq_shift
cfc_output = data[0:freq_shift.size]
#low pass data to remove double frequency term
#cfc_output = butter_lowpass_filter(cfc_output, 800e3, sample_rate)
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

def runpll(ffc_input, vco_p_gain, vco_i_gain):
    #initialize DPLL
    vco_curr_phase = 0
    vco_curr_freq = 0 #this is in hz - starting frequency
    vco_integrator = 0

    #phases for clusters
    cluster_phases = [0, np.pi/2, np.pi, 3*np.pi/2]

    #setup PLL output array
    pll_output_array = np.zeros(ffc_input.shape, dtype=np.complex64)
    error_array = np.zeros(ffc_input.shape, dtype=np.complex64)
    for index,sample in enumerate(ffc_input):
        #generate VCO signal
        #vco_curr_phase += vco_curr_freq * 2 * np.pi / sample_rate
        #vco_curr_phase = np.mod(vco_curr_phase, np.pi)
        #calculate phase-shifted input
        input_phase = np.angle(sample) + vco_curr_phase
        #store PLL output
        pll_output_array[index] = np.abs(sample) * np.exp(1j*input_phase)
        #phase comparison
        min_error = 2*np.pi
        for phase in cluster_phases:
            error = input_phase - phase
            while error > np.pi:
                error -= 2*np.pi
            while error < -1 * np.pi:
                error += 2*np.pi
            if np.abs(min_error) > np.abs(error):
                min_error = error

        error_array[index] = min_error
        vco_integrator += min_error
        #update VCO
        vco_curr_phase -= vco_p_gain * min_error + vco_i_gain * vco_integrator


    return pll_output_array, error_array

'''
for i in range(4, 8):
    output = runpll(cfc_output, 10**i)
    print(10**i)
    plt.figure()
    plt.title(f'PLL Output - gain={10**i}')
    plt.plot(timescale[6800:8000], np.angle(output[6800:8000]))
'''
packet_start = 8400
packet_end = 8800


pll_output_array, error_array = runpll(cfc_output[packet_start:packet_end], 0.4, 0)


plot_start = 0
plot_end = pll_output_array.size
'''
min_error_sum = np.sum(np.power(error_array[plot_start:plot_end],2))
best_params = (1,1)

for i in np.arange(0,2,0.1):
    for j in np.arange(0,2,0.1):
        print(f'running PLL with parameters {(i,j)}')
        pll_output_array, error_array = runpll(cfc_output, i, j)
        error_sum = np.sum(np.power(error_array[plot_start:plot_end],2))
        print(f'error sum: {error_sum}')
        if error_sum < min_error_sum:
            min_error_sum = error_sum
            best_params = (i, j)

print(f'best parameters: {best_params} with error sum of {min_error_sum}')
'''

print('plotting')
fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.title('Loop Filter Output')
ax.scatter3D(timescale[plot_start:plot_end], np.real(pll_output_array[plot_start:plot_end]), np.imag(pll_output_array[plot_start:plot_end]))
plt.figure()
plt.title('PLL Error')
plt.plot(timescale[plot_start:plot_end],error_array[plot_start:plot_end])
plt.figure()
plt.title('PLL Input - Phase')
plt.scatter(timescale[plot_start:plot_end],np.angle(cfc_output[packet_start:packet_end]))
zero_line = np.full(timescale[plot_start:plot_end].shape, 0)
pi_2_line = np.full(timescale[plot_start:plot_end].shape, np.pi/2)
pi_line = np.full(timescale[plot_start:plot_end].shape, np.pi)
neg_pi_2_line = np.full(timescale[plot_start:plot_end].shape, -np.pi/2)
neg_pi_line = np.full(timescale[plot_start:plot_end].shape, -np.pi)
plt.plot(timescale[plot_start:plot_end],zero_line)
plt.plot(timescale[plot_start:plot_end],pi_2_line)
plt.plot(timescale[plot_start:plot_end],pi_line)
plt.plot(timescale[plot_start:plot_end],neg_pi_2_line)
plt.plot(timescale[plot_start:plot_end],neg_pi_line)
plt.figure()
plt.title('PLL Output - Phase')
plt.scatter(timescale[plot_start:plot_end],np.angle(pll_output_array[plot_start:plot_end]))
plt.show()
