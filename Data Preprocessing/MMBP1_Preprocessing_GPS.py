import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import utm
import gc

'''   GPS   '''

#%% Create and save GPS
df_train =  pd.read_csv('./ml_challenge_dev_multi_modal.csv')

beamnum = df_train['unit1_beam'].values
index = df_train['index'].values
pos_rel_paths_1 = df_train['unit2_loc_1'].values
pos_rel_paths_2 = df_train['unit2_loc_2'].values
bs_rel_paths = df_train['unit1_loc'].values
n_samples = len(index)
pos_array_1 = np.zeros((n_samples, 2)) # 2 = Latitude and Longitude
pos_array_2 = np.zeros((n_samples, 2))
bs_array = np.zeros((n_samples, 2))
# Load each individual txt file
for sample_idx in range(n_samples):
    #pos_abs_path = os.path.join(scenario_folder, pos_rel_paths[sample_idx])
    pos_abs_path =  pos_rel_paths_1[sample_idx]
    pos_array_1[sample_idx] = np.loadtxt(pos_abs_path)
    pos_abs_path =  pos_rel_paths_2[sample_idx]
    pos_array_2[sample_idx] = np.loadtxt(pos_abs_path)
    bs_abs_path = bs_rel_paths[sample_idx]
    bs_array[sample_idx] = np.loadtxt(bs_abs_path)


#%% All raw GPS data
beam_label = beamnum -1

N_POS = 2 # num of coords
N_BEAMS = 64
n_samples = len(pos_array_1)
# no. input sequences x no. samples per seq x sample dimension (lat and lon)
pos_input = np.zeros((n_samples, 2, N_POS)) 
pos_input[:, 0, :] = pos_array_1 #time sequence 1
pos_input[:, 1, :] = pos_array_2 #time sequence 2
pos_bs = bs_array

plt.scatter(pos_input[:, :, 0], pos_input[:, :, 1]); #plt.scatter(pos_bs[:,0],pos_bs[:,1]); 
plt.title('GPS Position, not interpolated, not normalized'); plt.xlabel('X (latitude)'); 
plt.ylabel('Y (longitude)'); plt.legend(['user']); #plt.legend(['user','BS']); 
plt.show()

print(pos_input[:2,:,:])

#%% Extrapolate to have tensor of size (11143,5,2) instead of (11143,2,2)
# Initialize the extrapolated dataset
extrapolated_gps_data = np.zeros((11143, 5, 2))
extrapolated_gps_data[:, :2, :] = pos_input

# Iterate through each sample
for i in range(11143):
    # Extract the positions of the first two time sequences
    x0, y0 = pos_input[i, 0, 0], pos_input[i, 0, 1]  # Lat and Long for t-4
    x1, y1 = pos_input[i, 1, 0], pos_input[i, 1, 1]  # Lat and Long for t-3
    xdif = x1 - x0
    ydif = y1 - y0

    # Check if x1 is not equal to x0 to avoid division by zero
    if (x1 != x0) & (y1 != y0):
        # Calculate the slope (m) and y-intercept (b) for the line equation y = mx + b
        m = (y1 - y0) / (x1 - x0)
        b = y0 - m * x0 

        # Extrapolate the positions for the next three time sequences (t-2, t-1, t)
        for j in range(2, 5):
            # Calculate the latitude and longitude using the line equation
            extrapolated_gps_data[i, j, 0] = extrapolated_gps_data[i, j-1, 0] + xdif  # Extrapolated latitude
            extrapolated_gps_data[i, j, 1] = m * extrapolated_gps_data[i, j, 0] + b  # Extrapolated longitude

    elif (x1 == x0) & (y1 == y0):
        # Handle the case where x2 is equal to x1 (avoid division by zero)
        extrapolated_gps_data[i, 2:, :] = pos_input[i, 1, :]  # Set the next points to be the same as t-3

    else:
        for j in range(2, 5):
            extrapolated_gps_data[i, j, 0] = extrapolated_gps_data[i, j-1, 0] + xdif  # Extrapolated latitude
            extrapolated_gps_data[i, j, 1] = extrapolated_gps_data[i, j-1, 1] + ydif  # Extrapolated longitude

plt.scatter(extrapolated_gps_data[:, :, 0], extrapolated_gps_data[:, :, 1]); #plt.scatter(pos_bs[:,0],pos_bs[:,1]); 
plt.title('GPS Position, extrapolated, not normalized'); plt.xlabel('X (latitude)'); 
plt.ylabel('Y (longitude)'); plt.legend(['user']); #plt.legend(['user','BS']); 
plt.show()

plt.scatter(pos_input[:, :, 0], pos_input[:, :, 1]); #plt.scatter(pos_bs[:,0],pos_bs[:,1]); 
plt.title('GPS Position, not extrapolated, not normalized'); plt.xlabel('X (latitude)'); 
plt.ylabel('Y (longitude)'); plt.legend(['user']); #plt.legend(['user','BS']); 
plt.show()

gc.collect()

#%% Normalize positions (min-max XY position difference between UE and BS) 
def xy_from_latlong(lat_long):
    """
    Requires lat and long, in decimal degrees, in the 1st and 2nd columns.
    Returns same row vec/matrix on cartesian (XY) coords.
    """
    # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
    x, y, *_ = utm.from_latlon(lat_long[:,0], lat_long[:,1])
    return np.stack((x,y), axis=1)

pos_ue_stacked = np.vstack((extrapolated_gps_data[:, 0, :], extrapolated_gps_data[:, 1, :],\
     extrapolated_gps_data[:, 2, :], extrapolated_gps_data[:, 3, :], \
        extrapolated_gps_data[:, 4, :]))
pos_bs_stacked = np.vstack((pos_bs, pos_bs, pos_bs, pos_bs, pos_bs))

pos_ue_cart = xy_from_latlong(pos_ue_stacked)
pos_bs_cart = xy_from_latlong(pos_bs_stacked)

pos_diff = pos_ue_cart - pos_bs_cart

pos_min = np.min(pos_diff, axis=0)
pos_max = np.max(pos_diff, axis=0)

# Normalize and unstack
pos_stacked_normalized = (pos_diff - pos_min) / (pos_max - pos_min)
gps_data = np.zeros((n_samples, 5, 2))
gps_data[:, 0, :] = pos_stacked_normalized[:n_samples]
gps_data[:, 1, :] = pos_stacked_normalized[n_samples:2*n_samples]
gps_data[:, 2, :] = pos_stacked_normalized[2*n_samples:3*n_samples]
gps_data[:, 3, :] = pos_stacked_normalized[3*n_samples:4*n_samples]
gps_data[:, 4, :] = pos_stacked_normalized[4*n_samples:]

print('np.shape(gps_data) =', np.shape(gps_data))
print(gps_data[:2,:,:])

np.savez('GPS_11143x5x2.npz', GPS = gps_data)      

plt.scatter(gps_data[:, :, 0], gps_data[:, :, 1]); #plt.scatter(pos_bs[:,0],pos_bs[:,1]); 
plt.title('GPS Position, extrapolated, normalized'); plt.xlabel('X (latitude)'); 
plt.ylabel('Y (longitude)'); plt.legend(['user']); #plt.legend(['user','BS']); 
plt.show()

#%% Transform GPS to (11143,5,210,360)
data = np.load('GPS_11143x5x2.npz')
GPS = data['GPS']

max_lat = np.max(GPS[:, :, 0])
min_lat = np.min(GPS[:, :, 0])
max_lon = np.max(GPS[:, :, 1])
min_lon = np.min(GPS[:, :, 1])
rangeLat = max_lat - min_lat
rangeLon = max_lon - min_lon

GPS_transformed = np.zeros((11143,5,210,360))
for samp_idx in range(GPS.shape[0]): #iterate over every 11143 sample
    for time_seq in range(5):
        pointLat = GPS[samp_idx, time_seq, 0] #time sequence 0
        pointLon = GPS[samp_idx, time_seq, 1]
        GPS_transformed[samp_idx,time_seq,math.floor(((pointLon-min_lon)/rangeLon)*210)-1,math.floor((((pointLat-min_lat)/rangeLat)*360))-1] = 1        
        
np.sum(GPS_transformed) #confirming the total 11143x5 num of GPS position values existing in the transformed GPS data
np.savez('GPS_11143x5x210x360.npz', GPS = GPS_transformed) 

#%% Plot
gps_images = GPS_transformed[:,4,:,:]

# Combine all the GPS images into one array
combined_image = np.zeros((210, 360))
for gps_image in gps_images:
    combined_image += gps_image
np.sum(combined_image)
# Create a binary mask where locations with values greater than 0 are set to 1
mask = np.where(combined_image > 0, 1, 0)
np.sum(mask)

# Display the combined GPS locations image
plt.imshow(mask, cmap='hot', origin='upper')
plt.colorbar(label='Number of Locations')
plt.title('GPS Position at t-4, extrapolated, normalized, 2D form');
plt.show()