# Scaled-YOLOv4
cd /fs/home/k207198/h322330/ccc/PSD/Scaled-YOLOv4; conda activate PSD; python test.py --data /fs/home/k207198/h322330/ccc/data/mydata.yaml --weights runs/exp22_yolov4-p7aug2/weights/best.pt --task val --save-json --conf-thres 0.001 --iou-thres 0.5
YOLOR
cd /fs/home/k207198/h322330/ccc/PSD/YOLOR; conda activate PSD; python test.py --img 640  --batch 8 --device 0 --data /fs/home/k207198/h322330/ccc/mydata/mydata.yaml --cfg cfg/yolor_p6.cfg --weights runs/train/yolor_p6/weights/best.pt
# YOLOv5
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv5; conda activate PSD;python val.py --data /fs/home/k207198/h322330/ccc/data/mydata.yaml --weights runs/train/exp1/weights/best.pt --img 640
# YOLOv6 
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv6; conda activate PSD; python tools/eval.py --data /fs/home/k207198/h322330/ccc/data/mydata.yaml  --batch 32 --weights runs/train/v6n/weights/best_ckpt.pt --task val --reproduce_640_eval --verbose
# YOLOv7
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv7; conda activate PSD; python test.py --data /fs/home/k207198/h322330/ccc/mydata/mydata.yaml --img 640 --batch 32  --iou 0.5 --device 0 --weights runs/train/yolov7-custom3/weights/best.pt --name yolov7_640_val --verbose
# YOLOv8
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv8; conda activate PSD; yolo detect val model=runs/best.pt data=/fs/home/k207198/h322330/ccc/mydata/mydata.yaml
#YOLOv9 
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv8; conda activate PSD; python val_dual.py --data /fs/home/k207198/h322330/ccc/mydata/mydata.yaml --img 640 --batch 32 --conf 0.001 --iou 0.5 --device 0 --weights 'runs/train/yolov9-c/weights/best.pt' --save-json --name yolov9_e_640_val
