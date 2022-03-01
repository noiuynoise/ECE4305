from cmath import phase
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from helper_funcs import butter_lowpass_filter, butter_lowpass
from scipy import signal 
from scipy import integrate


sample_rate = 2e6 # Hz
center_freq = 2426e6 # Hz
fft_size = 2**10
modulation_index = 2
#load in sample data
data = np.fromfile('recorded_2MHz.iq', np.complex64)
timescale = np.arange(0, data.size / sample_rate, 1/sample_rate)
#frequency shift sampled data for testing
# data = data*np.exp(2j * np.pi * 250e3 * timescale)
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
cfc_output = data[0:freq_shift.size] #* freq_shift
#low pass data to remove double frequency term
# cfc_output = butter_lowpass_filter(cfc_output, 400e3, sample_rate)
area_list = []
mean_area_list = [0]
num_packets = 0
start_of_packet = []
end_of_packet = []
packet_present = False
#Integrator Packet Detecting
for i in range(len(data)):
    a = i
    b = i+20
    area = sum(abs(data[a:b])**2)
    area_list.append(area)
    mean_area = np.mean(area_list)
    end_flag = 0
    if mean_area > mean_area_list[-1] and packet_present == False:
        packet_present = True
        start_of_packet.append(i)
    if mean_area < mean_area_list[-1] and packet_present == True:
        packet_present = False
        end_flag = 1

    if end_flag == 1:
        num_packets += 1
        end_of_packet.append(i)

    mean_area_list.append(mean_area)

print(start_of_packet)
print(end_of_packet)
print(num_packets)

#Pruning False IDs
num_removed = 0
packet_buffer = 50
for i in range(len(start_of_packet)):
    if end_of_packet[i-num_removed]-start_of_packet[i-num_removed] < 200:
        del start_of_packet[i-num_removed]
        del end_of_packet[i-num_removed] 
        num_packets -= 1
        num_removed += 1
    else:
        start_of_packet[i-num_removed] = start_of_packet[i-num_removed] - packet_buffer
        end_of_packet[i-num_removed] = end_of_packet[i-num_removed] + packet_buffer
print(start_of_packet)
print(end_of_packet)
print(num_packets)

    
plt.figure()
plt.plot(area_list)
plt.title('area')
plt.figure()
plt.plot(mean_area_list)
plt.title('area')


start_samp = 19260
end_samp = 19580
ffc_input = cfc_output #[start_samp:end_samp]
phase_input = np.angle(ffc_input) 
plt.figure()
plt.plot(ffc_input)
plt.title('ffc_input')
plt.figure()
plt.plot(phase_input)
plt.title('phase_input')

#timescale2 = np.linspace(0, (ffc_input.size/10) / sample_rate, num=len(ffc_input/10))
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(np.arange(10), np.real(ffc_input[50:60]), np.imag(ffc_input[50:60]))


# print('running PLL')

# #FFC 
# b_coefs = signal.firls(31, [0, 500e3, 510e3, 1e6],[1, 1, 0, 0], fs=2e6)
# # high_b_co = signal.firls(19, [0, 2000, 2100, 1e6],[1, 1, 0, 0], fs=2e6)

# setup loop filter
# loop_filter_state = signal.lfilter_zi(b_coefs, [1])

#setup PLL output array
# pll_output_array = np.zeros(ffc_input.shape, dtype=np.complex64)
# loop_filter_input_array = np.zeros(ffc_input.shape, dtype=np.complex64)
# bias_output_array = np.zeros(ffc_input.shape, dtype=np.complex64)
# bias_input_array = np.zeros(ffc_input.shape, dtype=np.complex64)
# # New Error Calculations

# for index in range(len(phase_input)):
#     loop_filter_input_array[index] = phase_input[index]
#     loop_filter_output = signal.lfilter(b_coefs,[1],loop_filter_input_array[0:index+1])
#     # loop_filter_output, zf = signal.lfilter(b_coefs,[1], loop_filter_input_array[0:index+1], zi=loop_filter_state, axis=-1)
#     #store loop filter output
#     pll_output_array[index] = loop_filter_output[-1]

# phase_input = pll_output_array

# current_symbol = 0
# binary_out = []
# phase_dif_2 = []
# previous_phase = 0
# lower_decision_phase = np.pi/2
# upper_decision_phase = 2*np.pi-lower_decision_phase
# print(len(phase_input))
# for index in range(len(phase_input)):
#     current_phase = phase_input[index]
#     previous_phase = phase_input[index-1]
#     phase_dif = abs(abs(current_phase) - abs(previous_phase))
#     phase_dif_2.append(phase_dif)
#     if phase_dif > lower_decision_phase and phase_dif < upper_decision_phase:
#         symbol_change = True
#     else:
#         symbol_change = False

#     if symbol_change == True:
#         current_symbol = 1 if current_symbol == 0 else 0
#         binary_out.append(current_symbol)
#     else:
#         binary_out.append(current_symbol)
# plt.figure()
# plt.plot(np.arange(len(phase_dif_2)),phase_dif_2)
# plt.title('phase_dif')     
        
#     #generate VCO signal
#     # DFS_output = np.exp(1j*2*math.pi*2e6+error)
#     #mix VCO and input signals
#     # loop_filter_input = DFS_output * sample
    # loop_filter_input_array[index] = phase_input[index]
    # loop_filter_output = signal.lfilter(b_coefs,[1],loop_filter_input_array[0:index+1])
    # # loop_filter_output, zf = signal.lfilter(b_coefs,[1], loop_filter_input_array[0:index+1], zi=loop_filter_state, axis=-1)
    # #store loop filter output
    # pll_output_array[index] = loop_filter_output[-1]
#     bias_input_array[index] = loop_filter_output[-1]
#     bias_output = signal.lfilter(high_b_co,[1],bias_input_array[0:index+1])
#     bias_output_array[index] = bias_output[-1]
#     #update VCO

# plt.figure()
# plt.plot(pll_output_array)   
# plt.title('pll_output_array')    
# plt.figure()
# plt.plot(bias_output_array)   
# plt.title('bias_output_array')   
# plt.figure()
# plt.plot(pll_output_array-bias_output_array)   
# plt.title('PLL - bias')     
# ones = 0
# zeros = 0
# print(binary_out)
# print(len(binary_out))
# for i in range(int(len(binary_out)/2)):
#     sym_1 = binary_out[i*2]
#     sym_2 = binary_out[i*2+1]
#     if sym_1 == 1:
#         ones += 1
#     if sym_2 == 1:
#         ones += 1
#     if sym_1 == 0:
#         zeros += 1
#     if sym_2 == 0:
#         zeros += 1
#     bit = [sym_1,sym_2]
#     print(bit)
# print(ones)
# print(zeros)
# print('plotting')


# timescale2 = np.linspace(0, ffc_input.size / sample_rate, num=len(binary_out))
# plt.figure()
# plt.title('PLL Input')
# plt.plot(timescale2,np.imag(ffc_input)/np.amax(ffc_input))
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(timescale2, np.real(pll_output_array), np.imag(pll_output_array))
# plt.figure()
# plt.title('PLL Output')
# plt.plot(timescale2,np.imag(pll_output_array)/np.amax(pll_output_array))
# plt.figure()
# plt.title('PLL Output - Phase')
# plt.plot(timescale2,np.angle(pll_output_array))
# plt.figure()
# plt.plot(abs(np.fft.fftshift(np.fft.fft(pll_output_array))))
plt.show()

