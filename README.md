# MultiModal-Beam-Prediction-CNN-GRU-FCNN
Implementation of the [paper](https://ieeexplore.ieee.org/document/10636967)
# 1. Dataset 
Dataset is obtained from [DeepSense6G](https://www.deepsense6g.net/challenge2022/)
- We utilize Scenario 33-34 from the "Multi Modal Beam Prediction Challenge 2022" Challange
- The dataset consist of a sequence of 5 samples for Vision, LIDAR, and RADAR and 2 samples of GPS.
- The GPS samples are interpolated to obtain 5 samples. 
# 2. Data Preprocessing
Data Preprocessing Notebooks for each modality are locatined in the "Preprocessing" Folder
- Vision:
- Radar:
- LiDAR:
- GPS:
# CNN+GRU+FCNN
Train and Test single modality and fusion (early/late) Notebooks are located the "" Folder
