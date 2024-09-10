import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

'''   Radar   '''

def range_angle_map(data, fft_size = 64):
    data = np.fft.fft(data, axis = 1) # Range FFT
    data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data,fft_size,  axis = 0) # Angle FFT
    data = np.fft.fftshift(data,axes=0)
    data = np.abs(data).sum(axis = 2) # Sum over velocity
    return data.T

def range_velocity_map(data):
    data = np.fft.fft(data, axis=1) # Range FFT
    # data -= np.mean(data, 2, keepdims=True)
    data = np.fft.fft(data, axis=2) # Velocity FFT
    data = np.fft.fftshift(data, axes=2)
    data = np.abs(data).sum(axis = 0) # Sum over antennas
    data = np.log(1+data)
    return dat
df_train =  pd.read_csv('./ml_challenge_dev_multi_modal.csv')

def process_radar_data(df, num_paths=5):
    """Process radar data from multiple paths."""
    radar_data_all = []
    
    for i in range(1, num_paths + 1):
        radar_rel_paths = df[f'unit1_radar_{i}'].values
        # For range velocity map subsititute range_velocity_map function into the next line
        radar_temp = [range_angle_map(np.load(path)) for path in radar_rel_paths]
        radar_data_all.append(np.array(radar_temp))
        print(f'radar_{i} done...')
        print(f'radar_{i}.shape:', np.array(radar_temp).shape)
    
    # Combine all radar data into one array
    radar_combined = np.stack(radar_data_all, axis=1)  # This stacks along the second axis (axis=1)
    return radar_combined

# Processing the radar data
radar = process_radar_data(df_train)

# Optional: print out the shape to confirm it's correct
print(radar.shape)

sample_index = 0  # Change index to view different samples
plt.imshow(radar[sample_index, 0, :, :], aspect='auto')
plt.title('Sample Range-Angle Map')
plt.xlabel('Angle [$\degree$]')
plt.ylabel('Range [m]')
plt.colorbar()
plt.show()


