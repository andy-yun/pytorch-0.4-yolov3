# pytorch0.4-yolov3
## This repository is created for implmentation of yolov3 with pytorch 0.4 from marvis yolov2.

### Please refer to https://github.com/marvis/pytorch-yolo2 for the detail information.

### Train your own data

python train.py -d cfg/coco.data -c cfg/yolo_v3.cfg -w yolo_v3.weights

### detect the dog using pretrained weights

python detect.py cfg/yolo_v3.cfg yolo_v3.weights data/dog.jpg data/coco.names

![predictions](data/predictions.jpg)

Loading weights from model_dist\yolo_v3.weights... Done!

data\dog.jpg: Predicted in 0.837523 seconds.  
3 box(es) is(are) found  
dog: 0.999996  
truck: 0.995232  
bicycle: 0.999973  
save plot results to predictions.jpg  

### License

MIT License (see LICENSE file).
