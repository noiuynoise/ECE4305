import numpy as np
import adi
import matplotlib.pyplot as plt
import time

sample_rate = 10e6 # Hz
center_freq = 100e6 # Hz

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
#sdr.rx_lo = int(center_freq)
sdr.rx_buffer_size = 1024 # this is the buffer the Pluto uses to buffer samples
start_time = time.time()
for i in range(100):
    samples = sdr.rx() # receive samples off Pluto
end_time = time.time()
print('seconds elapsed:', end_time - start_time)
print('Connection Speed (samples/s):', 1/ (end_time - start_time) * 100 * 1024)