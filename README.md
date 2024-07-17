# PearSurfaceDefects Benchmark: Benchmarking the Deep Learning Techniques for Pear Surface Defects
Pear surface defect detection is one of the major challenges in pear production. Traditional defect detection methods rely on manual inspection, which is time-consuming, labor-intensive, and results in inconsistent detection outcomes. Machine vision (MV)-based defect detection offers a promising solution by identifying surface defects on pears through image recognition. Artificial intelligence (AI) powered by deep learning (DL) is emerging as a key driver in the development of MV technology for defect detection. To harness the potential of AI in defect detection, it is essential to systematically evaluate DL models and utilize a large-scale, accurately annotated dataset of pear surface defects.In this study, we introduce a novel benchmark, PearSurfaceDefects (PSD), for DL techniques in defect detection tasks within pear production systems. PSD is extensible, modular, and unified; it standardizes the defect recognition process by: 1) developing a scalable and diverse dataset, 2) modularizing DL implementations, and 3) unifying the evaluation protocol. A comprehensive benchmark of state-of-the-art deep learning algorithms for pear surface defect detection will be established. By leveraging the PSD pipeline, end users can readily focus on developing robust deep learning models with automated data processing and experimental evaluations. The dataset and source codes will be made publicly available.

## 1. Installation
* Create a conda environment:  `conda create -n PSD python=3.8 -y`   
* ctive the virtul environment: `conda activate PSD`  
* Install requirements: `pip install -r requirements.txt`  

## 2. Preparing the Dataset
### 2.1 Dataset Preparation
* Download the dataset from [here](https://drive.google.com/drive/folders/1INRSHXMOqBf39mobRi6iKkm01JjMyVgK?usp=drive_link), and place it in the datasets/ folder.   
* Run the script to convert the labeled data into YOLO format: `python commons/json2yolov.py`(The code includes dataset partitioning, you just need to change the paths)  
* (Optional) Run the script to convert the labeled data into VOC format: `python commons/json2coco.py `  
### 2.2 (Optional) Data Augmentation
* To augment the trianing images, one can refer to `Data_augmentation/Data_aug.py`  
* Sample images are also provided in `datasets/Data_aug/`, one can use `Data_augmentation/Data_demo_aug.py `to generate examples.   
### 2.3 (Optiona2) Dataset Analysis
* To analysis the dataset, we can run: `python commons/dataset_analysis.py`.  
### 2.4 (Optiona3) Prepare your Own Dataset
* Label the dataset using [Labelme](https://github.com/labelmeai/labelme).  
* Then transfer the dataset using the above two steps to convert the dataset to YOLO or COCO formats.  
## 3. Training and Testing
* Download the pretrained models from [here](https://drive.google.com/drive/folders/1DcrluarBcoHd0GLfDZXWvrIl0Dxzmy8V?usp=drive_link). and unzip to corresponding folders. For example, you need to put the yolov5n.pt, yolov5s.pt, yolov5m.pt, yolov5l.pt and yolov5x.pt under the YOLOv5/ folder.  
* We trained the models  on 8 GPUs, readers are recommended to look at the train_cuda_8.sh files. To train the models, we can run:  `bash -i train_cuda_8.sh.  `   
* To test the models, we can run: ` bash -i test.sh.   `  
## 4. Performance
The YOLO algorithms[1-7] used for our experiments are not maintained by us, please give credit to the authors of the YOLO algorithms[1-7].   
## Video Demos
The video demos can be accessed at [Video Demos](https://drive.google.com/file/d/1Vnq8XsbiG-iMOnVQS6wEAh-rO5Lu_Tjh/view?usp=drive_link)
## Reference
* [1-1] YOLOv5:None.  
* [1-2] YOLOv5:Implementation: https://github.com/ultralytics/yolov5.  
* [2-1] YOLOv6: Li, C., Li, L., Jiang, H., Weng, K., Geng, Y., Li, L., Ke., Z., Li, Q., Cheng, M., Nie, W., Li, Y.,Zhang, B., Liang, Y., Zhou, L., Xu, X., Chu, X., Wei, X., Wei, X. (2022a). YOLOv6: a single-stage object detection framework for industrial applications.https://arxiv.org/pdf/2209.02976.pdf.  
* [2-2] YOLOv6 Implementation: https://github.com/meituan/YOLOv6.  
* [3-1] YOLOv7: Wang, C.Y., Bochkovskiy, A., Liao, H.Y.M. (2022). YOLOv7: trainable bag-of-freebies sets new state-of-the-art for real-time object detectors.https://arxiv.org/pdf/2207.02696.pdf.   
* [3-2] YOLOv7 Implementation: https://github.com/WongKinYiu/yolov7.   
* [4-1] YOLOv8:None. 
* [4-2] YOLOv8 Implementation: https://github.com/ultralytics/ultralytics.
* [5-1]  Wang, C. Y., Yeh, I. H., & Liao, H. Y. M. (2024). Yolov9: Learning what you want to learn using programmable gradient information.https://doi.org/10.48550/arXiv.2402.13616.  
* [5-2] YOLOv9 Implementation: https://github.com/WongKinYiu/yolov9.  
* [6-1] YOLOR: Wang, Chien-Yao, I-Hau Yeh, and Hong-Yuan Mark Liao. "You Only Learn One Representation: Unified Network for Multiple Tasks." arXiv preprint arXiv:2105.04206 (2021).   
* [6-2] YOLOR Implementation: https://github.com/WongKinYiu/yolor.  
* [7-1] ScaledYOLOv4: Wang, Chien-Yao, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. "Scaled-yolov4: Scaling cross stage partial network." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.  
* [7-2] ScaledYOLOv4 Implementation: https://github.com/WongKinYiu/ScaledYOLOv4.  
