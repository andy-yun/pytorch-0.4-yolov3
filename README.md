# pytorch-0.4-yolov3
## This repository is created for implmentation of yolov3 with pytorch 0.4 from marvis yolov2.

### Difference between this repository and marvis original version.

* some programs are re-structured for windows environments. (for example __name__ is always compared for starting).
* load and save weights are modified to compatible to yolov2 and yolov3 versions (means that this repository works for yolov2 and yolov3 configuration without source modification.)
* fully support yolov3 detction and training
   * region_loss.py is renamed to region_layer.py.
   * outputs of region_layer.py and yolo_layer.py are enclosed for dictionary variables.     
* codes are modified to work on pytorch 0.4 and python3
* some codes are modified to speed up and easy readings.

### Please refer to https://github.com/marvis/pytorch-yolo2 for the detail information.

### Train your own data
```
python train.py -d cfg/coco.data -c cfg/yolo_v3.cfg -w yolo_v3.weights
```

### detect the dog using pretrained weights

#### yolov2 models
```
wget http://pjreddie.com/media/files/yolo.weights
python detect.py cfg/yolo.cfg yolo.weights data/dog.jpg data/voc.names 
```

![predictions](data/predictions-yolov2.jpg)

Loading weights from yolo.weights... Done!  
data\dog.jpg: Predicted in 0.832918 seconds.  
3 box(es) is(are) found  
cat: 0.934710  
bicycle: 0.998012  
sheep: 0.990524  
save plot results to predictions.jpg  

#### yolov3 models
```
wget https://pjreddie.com/media/files/yolov3.weights
python detect.py cfg/yolo_v3.cfg yolov3.weights data/dog.jpg data/coco.names  
```

* detct.py is not changed much.

![predictions](data/predictions-yolov3.jpg)

Loading weights from yolov3.weights... Done!

data\dog.jpg: Predicted in 0.837523 seconds.  
3 box(es) is(are) found  
dog: 0.999996  
truck: 0.995232  
bicycle: 0.999973  
save plot results to predictions.jpg  



### License

MIT License (see LICENSE file).


