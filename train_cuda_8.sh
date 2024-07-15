# YOLOv5
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv5; conda activate PSD; python -m torch.distributed.run --nproc_per_node 8 train.py --batch 120 --data=/fs/home/k207198/h322330/ccc/data/mydata.yaml --cfg models/yolov5n.yaml --weights yolov5n.pt --device 0,1,2,3,4,5,6,7 --epochs 100
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv5; conda activate PSD; python -m torch.distributed.run --nproc_per_node 8 train.py --batch 120 --data=/fs/home/k207198/h322330/ccc/data/mydata.yaml --cfg models/yolov5s.yaml --weights yolov5n.pt --device 0,1,2,3,4,5,6,7 --epochs 100
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv5; conda activate PSD; python -m torch.distributed.run --nproc_per_node 8 train.py --batch 120 --data=/fs/home/k207198/h322330/ccc/data/mydata.yaml --cfg models/yolov5n.yaml --weights yolov5m.pt --device 0,1,2,3,4,5,6,7 --epochs 100
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv5; conda activate PSD; python -m torch.distributed.run --nproc_per_node 8 train.py --batch 120 --data=/fs/home/k207198/h322330/ccc/data/mydata.yaml --cfg models/yolov5n.yaml --weights yolov5l.pt --device 0,1,2,3,4,5,6,7 --epochs 100
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv5; conda activate PSD; python -m torch.distributed.run --nproc_per_node 8 train.py --batch 120 --data=/fs/home/k207198/h322330/ccc/data/mydata.yaml --cfg models/yolov5n.yaml --weights yolov5x.pt --device 0,1,2,3,4,5,6,7 --epochs 100
# YOLOv6
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv6; conda activate PSD; python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 120 --conf configs/yolov6n_finetune.py --data /fs/home/k207198/h322330/ccc/data/mydata.yaml --fuse_ab --device 0,1,2,3,4,5,6,7 --epochs 100
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv6; conda activate PSD; python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 120 --conf configs/yolov6s_finetune.py --data /fs/home/k207198/h322330/ccc/data/mydata.yaml --fuse_ab --device 0,1,2,3,4,5,6,7 --epochs 100
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv6; conda activate PSD; python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 120 --conf configs/yolov6m_finetune.py --data /fs/home/k207198/h322330/ccc/data/mydata.yaml --fuse_ab --device 0,1,2,3,4,5,6,7 --epochs 100
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv6; conda activate PSD; python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 120 --conf configs/yolov6l_finetune.py --data /fs/home/k207198/h322330/ccc/data/mydata.yaml --fuse_ab --device 0,1,2,3,4,5,6,7 --epochs 100
# YOLOv7
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv7; conda activate PSD; python -m torch.distributed.launch --nproc_per_node=8 train_aux.py --data /fs/home/k207198/h322330/ccc/data/mydata.yaml --weights 'yolov7_training.pt' --cfg cfg/training/yolov7.yaml --batch-size 120 --epochs 100  --device 0,1,2,3,4,5,6,7  --hyp data/hyp.scratch.custom.yaml
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv7; conda activate PSD; python -m torch.distributed.launch --nproc_per_node=8 train_aux.py --data /fs/home/k207198/h322330/ccc/data/mydata.yaml --weights 'yolov7x_training.pt' --cfg cfg/training/yolov7x.yaml --batch-size 120 --epochs 100  --device 0,1,2,3,4,5,6,7  --hyp data/hyp.scratch.custom.yaml
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv7; conda activate PSD; python -m torch.distributed.launch --nproc_per_node=8 train_aux.py --data /fs/home/k207198/h322330/ccc/data/mydata.yaml --weights 'yolov7-w6_training.pt' --cfg cfg/training/yolov7-w6.yaml --batch-size 120 --epochs 100  --device 0,1,2,3,4,5,6,7  --hyp data/hyp.scratch.custom.yaml
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv7; conda activate PSD; python -m torch.distributed.launch --nproc_per_node=8 train_aux.py --data /fs/home/k207198/h322330/ccc/data/mydata.yaml --weights 'yolov7-e6_training.pt' --cfg cfg/training/yolov7-e6.yaml --batch-size 120 --epochs 100  --device 0,1,2,3,4,5,6,7  --hyp data/hyp.scratch.custom.yaml
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv7; conda activate PSD; python -m torch.distributed.launch --nproc_per_node=8 train_aux.py --data /fs/home/k207198/h322330/ccc/data/mydata.yaml --weights 'yolov7-d6_training.pt' --cfg cfg/training/yolov7-d6.yaml --batch-size 120 --epochs 100  --device 0,1,2,3,4,5,6,7  --hyp data/hyp.scratch.custom.yaml
# YOLOv8
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv8; conda activate PSD; yolo detect train data=/fs/home/k207198/h322330/ccc/data/mydata.yaml model=yolov8n.pt epochs=100 batch=120  device=0,1,2,3,4,5,6,7
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv8; conda activate PSD; yolo detect train data=/fs/home/k207198/h322330/ccc/data/mydata.yaml model=yolov8s.pt epochs=100 batch=120  device=0,1,2,3,4,5,6,7
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv8; conda activate PSD; yolo detect train data=/fs/home/k207198/h322330/ccc/data/mydata.yaml model=yolov8m.pt epochs=100 batch=120  device=0,1,2,3,4,5,6,7
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv8; conda activate PSD; yolo detect train data=/fs/home/k207198/h322330/ccc/data/mydata.yaml model=yolov8l.pt epochs=100 batch=120  device=0,1,2,3,4,5,6,7
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv8; conda activate PSD; yolo detect train data=/fs/home/k207198/h322330/ccc/data/mydata.yaml model=yolov8x.pt epochs=100 batch=120  device=0,1,2,3,4,5,6,7

# YOLOv9
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv9; conda activate PSD; python -m torch.distributed.launch --nproc_per_node 8 --use_env --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 120 --data=/fs/home/k207198/h322330/ccc/mydata/mydata.yaml  --cfg=./models/detect/yolov9-c.yaml --weights=./yolov9-c.pt --hyp=hyp.scratch-high.yaml --epoch=100
cd /fs/home/k207198/h322330/ccc/PSD/YOLOv9; conda activate PSD; python -m torch.distributed.launch --nproc_per_node 8 --use_env --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 120 --data=/fs/home/k207198/h322330/ccc/mydata/mydata.yaml  --cfg=./models/detect/yolov9-e.yaml --weights=./yolov9-e.pt --hyp=hyp.scratch-high.yaml --epoch=100
