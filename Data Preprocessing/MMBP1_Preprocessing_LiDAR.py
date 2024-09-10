'''   LiDAR   '''

import gc
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import open3d as o3d
from plyfile import PlyData
from scipy.interpolate import interp1d
import math

#%% Create Lidar train dataset (intensity and distance)
print('Creating tensor started...')

def rescale_vector(vector, new_size):
    y = np.arange(len(vector))
    f = interp1d(y, vector, kind='linear')
    new_y = np.linspace(0, len(vector) - 1, new_size)
    rescaled_vector = f(new_y)
    return rescaled_vector

df_train =  pd.read_csv('./ml_challenge_dev_multi_modal.csv')
address = df_train['unit1_lidar_1'].values

num_seq = 5
new_size = 19000
num_features = 3 #angle, dist, inten

Lidar = np.zeros([len(address),num_seq,new_size,num_features])

address = df_train['unit1_lidar_1'].values
for i in range(len(address)): #iterate over all points in one point cloud
    cloud = o3d.io.read_point_cloud(address[i]) #take one point cloud
    xyz_load = np.asarray(cloud.points)
    plydata = PlyData.read(address[i])
    inten = np.asarray(plydata['vertex']['intensity']) #store intensity separately
    a = []
    d = []
    for index in range(len(xyz_load)):
        #calculate dist and angle from xyz
        ai = math.degrees(math.atan2(xyz_load[index,1],xyz_load[index,0]))
        # Adjust the range to 0-360 degrees
        if ai < 0:
            ai += 360
        di = np.sqrt(xyz_load[index,0]*xyz_load[index,0] + xyz_load[index,1]*xyz_load[index,1]+xyz_load[index,2]*xyz_load[index,2])
        if di < 0.25: #to zero the impact of zero-distance points in one point cloud
            di = 0
            inten[index] = 0
        #ai = np.arctan(xyz_load[index,1]/xyz_load[index,0])
        a.append(ai)
        d.append(di)
    Lidar[i,0,:,0] = rescale_vector(np.asarray(a), new_size)
    Lidar[i,0,:,1] = rescale_vector(np.asarray(d), new_size)
    Lidar[i,0,:,2] = rescale_vector(inten, new_size)

print('lidar1 done')

address = df_train['unit1_lidar_2'].values
for i in range(len(address)): #iterate over all points in one point cloud
    cloud = o3d.io.read_point_cloud(address[i]) #take one point cloud
    xyz_load = np.asarray(cloud.points)
    plydata = PlyData.read(address[i])
    inten = np.asarray(plydata['vertex']['intensity']) #store intensity separately
    a = []
    d = []
    for index in range(len(xyz_load)):
        #calculate dist and angle from xyz
        ai = math.degrees(math.atan2(xyz_load[index,1],xyz_load[index,0]))
        # Adjust the range to 0-360 degrees
        if ai < 0:
            ai += 360
        di = np.sqrt(xyz_load[index,0]*xyz_load[index,0] + xyz_load[index,1]*xyz_load[index,1]+xyz_load[index,2]*xyz_load[index,2])
        #ai = np.arctan(xyz_load[index,1]/xyz_load[index,0])
        a.append(ai)
        d.append(di)
    Lidar[i,1,:,0] = rescale_vector(np.asarray(a), new_size)
    Lidar[i,1,:,1] = rescale_vector(np.asarray(d), new_size)
    Lidar[i,1,:,2] = rescale_vector(inten, new_size)

print('lidar2 done')
    
address = df_train['unit1_lidar_3'].values
for i in range(len(address)): #iterate over all points in one point cloud
    cloud = o3d.io.read_point_cloud(address[i]) #take one point cloud
    xyz_load = np.asarray(cloud.points)
    plydata = PlyData.read(address[i])
    inten = np.asarray(plydata['vertex']['intensity']) #store intensity separately
    a = []
    d = []
    for index in range(len(xyz_load)):
        #calculate dist and angle from xyz
        ai = math.degrees(math.atan2(xyz_load[index,1],xyz_load[index,0]))
        # Adjust the range to 0-360 degrees
        if ai < 0:
            ai += 360
        di = np.sqrt(xyz_load[index,0]*xyz_load[index,0] + xyz_load[index,1]*xyz_load[index,1]+xyz_load[index,2]*xyz_load[index,2])
        #ai = np.arctan(xyz_load[index,1]/xyz_load[index,0])
        a.append(ai)
        d.append(di)
    Lidar[i,2,:,0] = rescale_vector(np.asarray(a), new_size)
    Lidar[i,2,:,1] = rescale_vector(np.asarray(d), new_size)
    Lidar[i,2,:,2] = rescale_vector(inten, new_size)

print('lidar3 done')

address = df_train['unit1_lidar_4'].values
for i in range(len(address)): #iterate over all points in one point cloud
    cloud = o3d.io.read_point_cloud(address[i]) #take one point cloud
    xyz_load = np.asarray(cloud.points)
    plydata = PlyData.read(address[i])
    inten = np.asarray(plydata['vertex']['intensity']) #store intensity separately
    a = []
    d = []
    for index in range(len(xyz_load)):
        #calculate dist and angle from xyz
        ai = math.degrees(math.atan2(xyz_load[index,1],xyz_load[index,0]))
        # Adjust the range to 0-360 degrees
        if ai < 0:
            ai += 360
        di = np.sqrt(xyz_load[index,0]*xyz_load[index,0] + xyz_load[index,1]*xyz_load[index,1]+xyz_load[index,2]*xyz_load[index,2])
        #ai = np.arctan(xyz_load[index,1]/xyz_load[index,0])
        a.append(ai)
        d.append(di)
    Lidar[i,3,:,0] = rescale_vector(np.asarray(a), new_size)
    Lidar[i,3,:,1] = rescale_vector(np.asarray(d), new_size)
    Lidar[i,3,:,2] = rescale_vector(inten, new_size)

print('lidar4 done')

address = df_train['unit1_lidar_5'].values
for i in range(len(address)): #iterate over all points in one point cloud
    cloud = o3d.io.read_point_cloud(address[i]) #take one point cloud
    xyz_load = np.asarray(cloud.points)
    plydata = PlyData.read(address[i])
    inten = np.asarray(plydata['vertex']['intensity']) #store intensity separately
    a = []
    d = []
    for index in range(len(xyz_load)):
        #calculate dist and angle from xyz
        ai = math.degrees(math.atan2(xyz_load[index,1],xyz_load[index,0]))
        # Adjust the range to 0-360 degrees
        if ai < 0:
            ai += 360
        di = np.sqrt(xyz_load[index,0]*xyz_load[index,0] + xyz_load[index,1]*xyz_load[index,1]+xyz_load[index,2]*xyz_load[index,2])
        #ai = np.arctan(xyz_load[index,1]/xyz_load[index,0])
        a.append(ai)
        d.append(di)
    Lidar[i,4,:,0] = rescale_vector(np.asarray(a), new_size)
    Lidar[i,4,:,1] = rescale_vector(np.asarray(d), new_size)
    Lidar[i,4,:,2] = rescale_vector(inten, new_size)

print('lidar5 done')

plt.figure();
plt.plot(Lidar[0, 0, :, 0], marker='.', markersize=2, linestyle='--', color='b') #Do it for angle, distance, intensity by changing Lidar column number here
plt.xlabel('components',fontsize=12)
plt.ylabel('Angle',fontsize=23)
plt.title('Angle variation in one Lidar measurement',fontsize=12)
plt.grid()
plt.show()

plt.figure();
plt.plot(Lidar[0, 0, :, 1], marker='.', markersize=2, linestyle='--', color='b') #Do it for angle, distance, intensity by changing Lidar column number here
plt.xlabel('components',fontsize=12)
plt.ylabel('Distance',fontsize=23)
plt.title('Distance variation in one Lidar measurement',fontsize=12)
plt.grid()
plt.show()

plt.figure();
plt.plot(Lidar[0, 0, :, 2], marker='.', markersize=2, linestyle='--', color='b') #Do it for angle, distance, intensity by changing Lidar column number here
plt.xlabel('components',fontsize=12)
plt.ylabel('Intensity',fontsize=23)
plt.title('Intensity variation in one Lidar measurement',fontsize=12)
plt.grid()
plt.show()

#%% zero the distance and intensity of small-distance points
for i in range(Lidar.shape[0]): #iterate over all points in one point cloud
    for j in range(Lidar.shape[1]):
        for k in range(Lidar.shape[2]):
            if Lidar[i, j, k, 1] < 0.5:
                Lidar[i, j, k, 1] = 0
                Lidar[i, j, k, 2] = 0
        
#%% Sort based on angle (step1)
print('Sort based on angle (step1) started...')

# Verify the shape and check the first few elements
print("Shape of dataset:", Lidar.shape)
print("Dataset (Lidar):")
print(Lidar[0, 0, :10, :])  # Printing first 10 elements for the first sample

plt.figure();
plt.plot(Lidar[0, 0, :, 0], marker='.', markersize=2, linestyle='--', color='b') #Do it for angle, distance, intensity by changing Lidar column number here
plt.xlabel('components',fontsize=12)
plt.ylabel('Angle',fontsize=23)
plt.title('Angle variation in one Lidar measurement, original',fontsize=12)
plt.grid()
plt.show()

plt.figure();
plt.plot(Lidar[0, 0, :, 1], marker='.', markersize=2, linestyle='--', color='b') #Do it for angle, distance, intensity by changing Lidar column number here
plt.xlabel('components',fontsize=12)
plt.ylabel('Distance',fontsize=23)
plt.title('Distance variation in one Lidar measurement, original',fontsize=12)
plt.grid()
plt.show()

plt.figure();
plt.plot(Lidar[0, 0, :, 2], marker='.', markersize=2, linestyle='--', color='b') #Do it for angle, distance, intensity by changing Lidar column number here
plt.xlabel('components',fontsize=12)
plt.ylabel('Intensity',fontsize=23)
plt.title('Intensity variation in one Lidar measurement, original',fontsize=12)
plt.grid()
plt.show()

angles = Lidar[:, :, :, 0]

# Reshape angles to match the shape of sorted_indices
angles_reshaped = np.expand_dims(angles, axis=-1)
print('angles_reshaped.shape =', angles_reshaped.shape)

# Get the sorted indices based on the "angle" values
sorted_indices = np.argsort(angles_reshaped, axis=2)
print('sorted_indices.shape =', sorted_indices.shape)

# Use the sorted indices to sort the entire dataset
Lidar = np.take_along_axis(Lidar, sorted_indices, axis=2)

# Verify the shape and check the first few elements
print("Shape of sorted dataset:", Lidar.shape)
print("Sorted dataset (Lidar):")
print(Lidar[0, 0, :10, :])  # Printing first 10 elements for the first sample

gc.collect()

plt.figure();
plt.plot(Lidar[2, 4, :, 0], marker='.', markersize=2, linestyle='--', color='b') #Do it for angle, distance, intensity by changing Lidar column number here
plt.xlabel('components',fontsize=12)
plt.ylabel('Angle',fontsize=23)
plt.title('Angle variation in one Lidar measurement, sorted',fontsize=12)
plt.grid()
plt.show()

plt.figure();
plt.plot(Lidar[2, 4, :, 1], marker='.', markersize=2, linestyle='--', color='b') #Do it for angle, distance, intensity by changing Lidar column number here
plt.xlabel('components',fontsize=12)
plt.ylabel('Distance',fontsize=23)
plt.title('Distance variation in one Lidar measurement, sorted',fontsize=12)
plt.grid()
plt.show()

plt.figure();
plt.plot(Lidar[2, 4, :, 2], marker='.', markersize=2, linestyle='--', color='b') #Do it for angle, distance, intensity by changing Lidar column number here
plt.xlabel('components',fontsize=12)
plt.ylabel('Intensity',fontsize=23)
plt.title('Intensity variation in one Lidar measurement, sorted',fontsize=12)
plt.grid()
plt.show()


#%% transform (11143, 5, 19000, 3) to (11143, 5, 210*k, 360*k)

gc.collect()

#find data range (enough to do once for all 5 time instances)
AngRange = np.ptp(Lidar[0, 0, :, 0])
DistRange = np.ptp(Lidar[0, 0, :, 1])
IntenRange = np.ptp(Lidar[0, 0, :, 2])
for samp_idx0 in range(Lidar.shape[0]):
    for samp_idx1 in range(Lidar.shape[1]):
        if AngRange <= np.ptp(Lidar[samp_idx0, samp_idx1, :, 0]):
            AngRange = np.ptp(Lidar[samp_idx0, samp_idx1, :, 0])
        if DistRange <= np.ptp(Lidar[samp_idx0, samp_idx1, :, 1]):
            DistRange = np.ptp(Lidar[samp_idx0, samp_idx1, :, 1])
        if IntenRange <= np.ptp(Lidar[samp_idx0, samp_idx1, :, 2]):
            IntenRange = np.ptp(Lidar[samp_idx0, samp_idx1, :, 2])
print('AngRange =', AngRange)        
print('DistRange =', DistRange)        
print('IntenRange =', IntenRange)    

k = 1 #quantization level coefficient
DistQuantLevelNum = np.ceil(DistRange) * k #num of quantized levels of distance values
AngQuantLevelNum = np.ceil(AngRange) * k #num of quantized levels of angle values
Lidar_transformedv2 = np.zeros([5, 1, int(DistQuantLevelNum), int(AngQuantLevelNum)]) #[11143, 210*k, 360*k]

for samp_idx in range(Lidar.shape[0]):
    for time_seq in range(5):
        Lidar_slice = Lidar[samp_idx, time_seq, :, :]
        AngQuantLevelNum_idx = np.arange(AngQuantLevelNum)
        DistQuantLevelNum_idx = np.arange(DistQuantLevelNum)

        sameCell_idx = (Lidar_slice[:, 0, np.newaxis] >= AngQuantLevelNum_idx / k) & \
                (Lidar_slice[:, 0, np.newaxis] < (AngQuantLevelNum_idx + 1) / k) & \
                (Lidar_slice[:, 1, np.newaxis] >= DistQuantLevelNum_idx / k) & \
                (Lidar_slice[:, 1, np.newaxis] < (DistQuantLevelNum_idx + 1) / k)

        aveInten = np.mean(Lidar_slice[sameCell_idx], axis=0)
        Lidar_transformedv2[samp_idx, 0, DistQuantLevelNum_idx, AngQuantLevelNum_idx] = aveInten

        Lidar_transformedv2[samp_idx, 0, ~sameCell_idx] = 0

    if samp_idx % 50 == 0:
        print("samp_idx:", samp_idx)

#np.savez('Lidar_transformed_Viterbi_timeSeq2_11143x1x210x225_intenAve.npz', Lidar_transformedv2=Lidar_transformedv2)

plt.imshow(Lidar_transformedv2[0, 0, :, :])
plt.xlabel("Angle [0 360]")
plt.ylabel("Distance [0 210]")
plt.title('sample # 0')
plt.colorbar()
plt.show()


#%% Bilateral Filter

def get_range(vector):
    min_value = np.min(vector)
    max_value = np.max(vector)
    value_range = max_value - min_value
    return min_value, max_value, value_range

lidar = np.copy(Lidar_transformedv2)
del Lidar_transformedv2

print('lidar.shape =', lidar.shape)
print("Data type:", lidar.dtype)

min_value, max_value, value_range = get_range(lidar[0,0,:,:])
print("Min, MAx, Range of z values:", min_value, max_value, value_range)

img_width = 210  # Width of the image plane
img_height = 360  # Height of the image plane
#factor = 1.0  # Intensity enhancement factor for visualization

filteredLidars = np.zeros((lidar.shape[0], lidar.shape[1], img_width, img_height))

#bilateral filter params
d = 9 #Diameter of each pixel neighborhood
sigma_color = 75 #The greater, the colors farther to each other will start to get mixed
sigma_space = 75 #The greater, the more further pixels will mix together, given that their colors lie within the sigmaColor range

for samp_idx in range(lidar.shape[0]):
    for seq_idx in range(lidar.shape[1]):               
        # Apply bilateral filter
        filteredLidar = cv2.bilateralFilter(lidar[samp_idx,seq_idx,:,:], d, sigma_color, sigma_space, borderType=cv2.BORDER_CONSTANT)
        filteredLidars[samp_idx,seq_idx,:,:] = filteredLidar
        
#np.savez('lidar_DepthInten_11143x5x210x360_v2.npz', Lidar=filteredLidars)
gc.collect()

print('filteredLidars.shape =', filteredLidars.shape)

# Display the Lidar point cloud in pixel coordinates
plt.imshow(filteredLidars[0,0,:,:])
plt.title("filtered Lidar")
plt.axis('off')  # Remove the axis labels
plt.show()