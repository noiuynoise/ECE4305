import numpy as np
import math
import matplotlib.pyplot as plt
from helper_funcs import butter_lowpass_filter, butter_lowpass
from scipy import signal 


sample_rate = 1e6 # Hz
center_freq = 2480e6 # Hz
fft_size = 2**14
modulation_index = 0.5
#load in sample data
data = np.fromfile('recorded_good_3.iq', np.complex64)
timescale = np.arange(0, data.size / sample_rate, 1/sample_rate)
print('running CFC')
#raised cosine / low pass filter
#TODO

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
    bin_sum = np.sum(np.abs(fft_bins))
    curr_sum = 0
    curr_index = 0
    while(curr_sum < bin_sum/2):
        curr_sum += np.abs(fft_bins[curr_index])
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
    vco_curr_freq = -1 * freq_init #this is in hz - starting frequency
    vco_integrator = 0

    #phases for clusters
    #cluster_phases = [np.pi/2, -1*np.pi/2]
    #cluster_phases = [0, np.pi, -1*np.pi]
    cluster_phases = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
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
        vco_curr_freq -= vco_p_gain * min_error


    print(f'end FFC frequency: {vco_curr_freq}')

    return pll_output_array, error_array



packet_to_decode = corrected_packets[0]
packet_avg_power = np.sum(np.power(np.abs(corrected_packets[0][2]), 2))/corrected_packets[0][2].size
for index, packet in enumerate(corrected_packets):    
    new_packet_avg_power = np.sum(np.power(np.abs(packet[2]), 2))/packet[2].size
    print(f'found packet {index} with length {packet[1] - packet[0]} and avg power {new_packet_avg_power}')

    if new_packet_avg_power > packet_avg_power:
        packet_to_decode = packet
        packet_avg_power = new_packet_avg_power
        print(f'packet{index} is more powerful with avg power {new_packet_avg_power}')

packet_start = 0
packet_end = packet_to_decode[1] - packet_to_decode[0]



cfc_output = packet_to_decode[2]

pll_output_array, error_array = runpll(cfc_output, 89125,0, packet_to_decode[3])


plot_start = 0
plot_end = packet_to_decode[1] - packet_to_decode[0] 

def try_pll(packet, kP, kI):
    pll_output_array, error_array = runpll(packet[2], kP, kI, packet[3])
    return np.sum(np.power(error_array,4))

tune_pll = False
if tune_pll:
    pll_output_array, error_array = runpll(cfc_output, 0,0, packet_to_decode[3])
    min_error_sum = np.sum(np.power(error_array[plot_start:plot_end],4))
    best_params = (0,0)

    #first scan on log scale
    scan_magnitude = 0
    for i in np.arange(-10, 6,0.5):
            print(f'running PLL with parameters {(10.0**i,0)}')
            error_sum = try_pll(packet_to_decode, 10.0**i, 0)
            print(f'error sum: {error_sum}')
            if error_sum < min_error_sum:
                min_error_sum = error_sum
                scan_magnitude = i
                best_params = (10.0**i, 0)
    
    #then try values in scale
    for i in np.arange(-1, 1, 0.01):
            print(f'running PLL with parameters {(10.0**(i+scan_magnitude),0)}')
            error_sum = try_pll(packet_to_decode, 10.0**(i+scan_magnitude), 0)
            print(f'error sum: {error_sum}')
            if error_sum < min_error_sum:
                min_error_sum = error_sum
                best_params = (10.0**(i+scan_magnitude), 0)

    print(f'best parameters: {best_params} with error sum of {min_error_sum}')

    pll_output_array, error_array = runpll(cfc_output, best_params[0], best_params[1], packet_to_decode[3])

#phase detector for bit detection
#the output should be oriented so decision boundary is imaginary axis. we can just read the real value for binary data
binary_data = np.zeros(pll_output_array.shape, dtype=np.byte)
for index, sample in enumerate(pll_output_array):
    if np.abs(np.real(sample)) > np.abs(np.imag(sample)):
        if np.real(sample) > 0:
            binary_data[index] = 0#2
    else:
        if np.imag(sample) > 0:
            binary_data[index] = 1
        else:
            binary_data[index] = 1#3

binary_data_delta = np.zeros(pll_output_array.shape, dtype=np.byte)
last_bit = binary_data[0]
for index, sample in enumerate(binary_data):
    if index == 0:
        continue
    if last_bit == sample:
        binary_data_delta[index] = 1
    last_bit = sample
#print(binary_data_delta)
#'''


print('plotting')
#plt.plot(np.abs(np.fft.fftshift(np.fft.fft(packet_to_decode[2]))))
#plt.title('input FFT')
#plt.figure()
#plt.plot(np.abs(np.fft.fftshift(np.fft.fft(pll_output_array))))
#plt.title('output FFT')
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
binary_data_len = binary_data_delta.size
binary_data_ones = np.sum(binary_data_delta)
print(f'{binary_data_len} bits detected with {binary_data_ones} ones, ratio of {binary_data_ones/binary_data_len}')
print(binary_data)
plt.show()
