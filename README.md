# MultiModal-Beam-Prediction-CNN-GRU-FCNN
Implementation of the [paper](https://ieeexplore.ieee.org/document/10636967)
# 1. Dataset 
Dataset is obtained from [DeepSense6G](https://www.deepsense6g.net/challenge2022/)
- We utilize Scenario 33-34 from the "Multi Modal Beam Prediction Challenge 2022" Challange
- The dataset consist of a sequence of 5 samples for Vision, LIDAR, and RADAR and 2 samples of GPS.
- The GPS samples are interpolated to obtain 5 samples. 
# 2. Data Preprocessing
Data Preprocessing and Data Visualization Notebooks for each modality are locatined in the "Data Preprocessing" Folder
- Radar:  "Preprocessing_Radar.py"
-   Conversion from raw Radar samples to Range-Angle maps or Range-Velocity Maps
- LiDAR: "Preprocessing_LiDAR.py"
- GPS: "Preprocessing_GPS.py"
- Vision: There is no special preprocessing for Vision. The RGB samples are converted to grayscale and then resized to (150,150)
# CNN+GRU+FCNN
Train and Test single modality and fusion (early/late) Notebooks are located the "CNN+GRU+FCNN Network" Folder
