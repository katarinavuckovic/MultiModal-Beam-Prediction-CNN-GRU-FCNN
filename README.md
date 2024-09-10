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
  - Conversion from raw Radar samples to Range-Angle maps or Range-Velocity Maps
- LiDAR: "Preprocessing_LiDAR.py"
  - Extraction of angle, distance and intesity features. 
- GPS: "Preprocessing_GPS.py"
  - Conversion to Cartesian coordinates with BS at the center. 
- Vision: There is no special preprocessing for Vision. The RGB samples are converted to grayscale and then resized to (150,150)
# 3. Train and Test Models
Train and Test single modality and fusion (early/late) Notebooks are located the "CNN+GRU+FCNN Network" Folder.
In the test section the following metrics are evaluated: top-3 accurac, DBA Score, top3 beam, Power Ratio (PR) (Same as Power Factor), Precision/Recall (P/R)
Single Modalities:
- GPS: "G_multipleRuns.ipynb"
- Radar: "R1_multipleRuns.ipynb"
- LiDAR: "L_multipleRuns.ipynb"
- Vision: "V_multipleRuns.ipynb"
Early Fusion:
-  Vision+GPS: "VG_multipleRuns.ipynb"
-  LiDAR+Vision+RADAR: "LVR_multipleRuns.ipynb"
-  LiDAR+Vision+RADAR+GPS: "LVRG_multipleRuns.ipynb"
Late Fusion:
- Vision+GPS: "LateFusion_VG_multipleRuns.ipynb"
- Vision+LiDAR+RADAR+GPS: "LateFusion_VLRG_multipleRuns.ipynb"
# 4. Reference
If you use this script or part of it, please cite the following:
 [paper](https://ieeexplore.ieee.org/document/10636967)
