import numpy as np
import math
import matplotlib.pyplot as plt
from helper_funcs import butter_lowpass_filter, butter_lowpass
from scipy import signal 


sample_rate = 1e6 # Hz
center_freq = 2426e6 # Hz
fft_size = 2**14
modulation_index = 0.5
#load in sample data
data = np.fromfile('recorded.iq', np.complex64)
timescale = np.arange(0, data.size / sample_rate, 1/sample_rate)
print('running CFC')
#raised cosine / low pass filter
#TODO
'''
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
#alternate band pass instead of low pass
N = 11
fp = 500e3
pw = 200e3
fc = 100e3
#filter = signal.firls(N, [0, fp-pw-fc, fp-pw, fp+pw, fp+pw+fc, sample_rate/2], [0, 0, 1, 1, 0, 0], fs=sample_rate)
#cfc_output = signal.lfilter(filter, 1, cfc_output)
'''
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

#find the start and end of the packet(s) - power based
rx_power = np.power(np.abs(data), 2)
#now low pass - 100khz should remove noise but not packet?
rx_power_lp = butter_lowpass_filter(rx_power, 100e3, sample_rate, order=3)
#if power > threshold set start and end of packet
#threshold is avg power?
avg_power = np.sum(rx_power)/rx_power.size
#plt.plot(timescale,rx_power_lp)
#plt.plot(timescale, np.full(rx_power_lp.shape, avg_power))
#now find all packets contained
packets = []
index = 0
while index < rx_power_lp.size:
    if rx_power_lp[index] > avg_power:
        #start of packet found
        start_sample = index - 20
        if start_sample < 0:
            start_sample = 0
        #now find end
        end_sample = rx_power_lp.size -1
        while index < end_sample:
            index += 1
            if rx_power_lp[index] < avg_power:
                #found end of packet
                end_sample = index + 20
                if end_sample > (rx_power_lp.size - 1):
                    end_sample = rx_power_lp.size -1
                break
        packets.append((start_sample, end_sample))
    index += 1

#CFC function
def cfc_packet(packet, samp_rate, mod_index):
    packet_no_modulation = np.power(packet, mod_index)
    freq_range = np.linspace(-samp_rate/2, samp_rate/2, num=packet_no_modulation.size)
    fft_bins = np.fft.fftshift(np.fft.fft(packet_no_modulation))
    bin_sum = np.sum(fft_bins)
    curr_sum = 0
    curr_index = 0
    while(curr_sum < bin_sum/2):
        curr_sum += fft_bins[curr_index]
        curr_index += 1
    freq_offset = freq_range[curr_index]
    #now shift signal
    freq_shift_timescale = np.arange(0, packet_no_modulation.size/samp_rate, 1/samp_rate)
    freq_shift_timescale = freq_shift_timescale[0:packet_no_modulation.size]
    print(f'frequency shifting by {freq_offset}')
    freq_shift = np.exp(-1j*np.pi*freq_offset*freq_shift_timescale)
    return freq_offset
#now perform CFC on all packets
corrected_packets = []
for packet in packets:
    corrected_packets.append((packet[0], packet[1],data[packet[0]:packet[1]], cfc_packet(data[packet[0]:packet[1]], sample_rate, modulation_index)))

#PLL function
def runpll(ffc_input, vco_p_gain, vco_i_gain, freq_init):
    #initialize DPLL
    vco_curr_phase = np.angle(ffc_input[0])
    vco_curr_freq = -0.5 * freq_init #this is in hz - starting frequency
    vco_integrator = 0

    #phases for clusters
    #cluster_phases = [np.pi/2, -1*np.pi/2]
    cluster_phases = [0, np.pi, -1*np.pi]
    #cluster_phases = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    #cluster_phases = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi]
    
    #setup PLL output array
    pll_output_array = np.zeros(ffc_input.shape, dtype=np.complex64)
    error_array = np.zeros(ffc_input.shape, dtype=np.complex64)
    for index,sample in enumerate(ffc_input):
        #generate VCO signal
        vco_curr_phase += vco_curr_freq * 2 * np.pi / sample_rate
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
        vco_curr_freq += vco_p_gain * min_error + vco_i_gain * vco_integrator


    return pll_output_array, error_array



packet_to_decode = corrected_packets[0]
packet_avg_power = np.sum(np.power(np.abs(corrected_packets[0][2]), 2))/corrected_packets[0][2].size
for index, packet in enumerate(corrected_packets):
    new_packet_avg_power = np.sum(np.power(np.abs(packet[2]), 2))/packet[2].size
    if new_packet_avg_power > packet_avg_power:
        packet_to_decode = packet
        packet_avg_power = new_packet_avg_power
        print(f'packet{index} is more powerful with avg power {new_packet_avg_power}')

packet_start = 0
packet_end = packet_to_decode[1] - packet_to_decode[0]



cfc_output = packet_to_decode[2]

pll_output_array, error_array = runpll(cfc_output, 2,0, packet_to_decode[3])


plot_start = 0
plot_end = packet_to_decode[1] - packet_to_decode[0] 
#'''
min_error_sum = np.sum(np.power(error_array[plot_start:plot_end],2))
best_params = (1.1,0.9)

for i in np.arange(0, 10,0.25):
    for j in np.arange(0, 1,0.2):
        print(f'running PLL with parameters {(i,j)}')
        pll_output_array, error_array = runpll(cfc_output, i, j, packet_to_decode[3])
        error_sum = np.sum(np.power(error_array,2))
        print(f'error sum: {error_sum}')
        if error_sum < min_error_sum:
            min_error_sum = error_sum
            best_params = (i, j)

print(f'best parameters: {best_params} with error sum of {min_error_sum}')

pll_output_array, error_array = runpll(cfc_output, best_params[0], best_params[1], packet_to_decode[3])

#phase detector for bit detection
#the output should be oriented so decision boundary is imaginary axis. we can just read the real value for binary data
binary_data = np.zeros(pll_output_array.shape, dtype=np.byte)
for index, sample in enumerate(pll_output_array):
    if np.real(sample) > 0:
        binary_data[index] = 1

#'''


print('plotting')
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Loop Filter Output')
ax.scatter3D(timescale[plot_start:plot_end], np.real(pll_output_array[plot_start:plot_end]), np.imag(pll_output_array[plot_start:plot_end]))
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Loop Filter Input')
ax.scatter3D(timescale[plot_start:plot_end], np.real(cfc_output), np.imag(cfc_output[packet_start:packet_end]))
#ax.scatter3D(timescale, np.real(cfc_output), np.imag(pll_output_array[plot_start:plot_end]))
plt.figure()
plt.title('PLL Error')
plt.plot(timescale[plot_start:plot_end],error_array[plot_start:plot_end])
plt.figure()
plt.title('PLL Input - Phase')
plt.plot(timescale[plot_start:plot_end],np.angle(cfc_output))
plt.plot(timescale[plot_start:plot_end],np.angle(pll_output_array[plot_start:plot_end]))
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
plt.plot(timescale[plot_start:plot_end],np.angle(pll_output_array[plot_start:plot_end]))
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
plt.title('output data')
plt.plot(timescale[plot_start:plot_end],binary_data)
binary_data_len = binary_data.size
binary_data_ones = np.sum(binary_data)
print(f'{binary_data_len} bits detected with {binary_data_ones} ones, ratio of {binary_data_ones/binary_data_len}')
plt.show()
