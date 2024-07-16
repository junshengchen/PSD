# PearSurfaceDefects Benchmark: Benchmarking the Deep Learning Techniques for Pear Surface Defects
Pear surface defect detection is one of the major challenges in pear production. Traditional defect detection methods rely on manual inspection, which is time-consuming, labor-intensive, and results in inconsistent detection outcomes. Machine vision (MV)-based defect detection offers a promising solution by identifying surface defects on pears through image recognition. Artificial intelligence (AI) powered by deep learning (DL) is emerging as a key driver in the development of MV technology for defect detection. To harness the potential of AI in defect detection, it is essential to systematically evaluate DL models and utilize a large-scale, accurately annotated dataset of pear surface defects.In this study, we introduce a novel benchmark, PearSurfaceDefects (PSD), for DL techniques in defect detection tasks within pear production systems. PSD is extensible, modular, and unified; it standardizes the defect recognition process by: 1) developing a scalable and diverse dataset, 2) modularizing DL implementations, and 3) unifying the evaluation protocol. A comprehensive benchmark of state-of-the-art deep learning algorithms for pear surface defect detection will be established. By leveraging the PSD pipeline, end users can readily focus on developing robust deep learning models with automated data processing and experimental evaluations. The dataset and source codes will be made publicly available.

## 1. Installation
Create a conda environment:  `conda create -n PSD python=3.8 -y`   
Active the virtul environment: `conda activate PSD`  
Install requirements: `pip install -r requirements.txt`  

## 2. Preparing the Dataset
### 2.1 Dataset Preparation
Download the dataset from [here](https://drive.google.com/drive/folders/1T9Pv6YMvOY6fOWoim21yEpgFuNg7dful?usp=drive_link), and place it in the datasets/ folder. 
Run the script to convert the labeled data into YOLO format: `python commons/json2yolov.py`(The code includes dataset partitioning, you just need to change the paths)  
(Optional) Run the script to convert the labeled data into VOC format: `python commons/json2coco.py `  
### 2.2 (Optional) Data Augmentation
To augment the trianing images, one can refer to `Data_augmentation/Data_aug.py`  
Sample images are also provided in `datasets/Data_aug/`, one can use `Data_augmentation/Data_demo_aug.py `to generate examples.   
### 2.3 (Optiona2) Dataset Analysis
To analysis the dataset, we can run: `python commons/dataset_analysis.py`.  
### 2.4 (Optiona3) Prepare your Own Dataset
Label the dataset using [Labelme](https://github.com/labelmeai/labelme).  
Then transfer the dataset using the above two steps to convert the dataset to YOLO or COCO formats.  
## 3. Training and Testing
Download the pretrained models from here and unzip to corresponding folders. For example, you need to put the yolov3.pt, yolov3-spp.pt and yolov3-tiny.pt under the YOLOV3/ folder.  
We trained the models for 5 replications on 5 GPUs, readers are recommended to look at the train_cudax.sh files. For instance, to run the 0st data folder, we can run: bash -i train_cuda0.sh.  
To test the models, we can run: bash -i test0.sh.  
## 4. Performance
The YOLO algorithms[1-6] used for our experiments are not maintained by us, please give credit to the authors of the YOLO algorithms[1-6].  
