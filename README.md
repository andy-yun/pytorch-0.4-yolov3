# pytorch0.4-yolov3
## This repository is created for implmentation of yolov3 with pytorch 0.4 from marvis yolov2.

### Please refer to https://github.com/marvis/pytorch-yolo2 for the detail information.

### Train your own data

python train.py -d cfg/coco.data -c cfg/yolo_v3.cfg -w yolo_v3.weights

### detect the dog using pretrained weights

python detect.py cfg/yolo_v3.cfg yolo_v3.weights data/dog.jpg data/coco.names

### License

MIT License (see LICENSE file).
