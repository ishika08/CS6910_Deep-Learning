<h2>Object Detection using YOLOv3</h2>

[Link to the wandb.ai report](https://wandb.ai/saish/Deep-Learning-CNN/reports/CS6910-Assignment-2---Vmlldzo2MDQ4NDA?accessToken=0d2a802xore8clx738gb2wuytbi54q9lyh6g4rlwxpt4bvs3d57qo3gc7uzisgzs)

**Usage**

```
usage: yolo.py [-h] [--webcam WEBCAM] [--play_video PLAY_VIDEO]
               [--image IMAGE] [--video_path VIDEO_PATH]
               [--image_path IMAGE_PATH] [--verbose VERBOSE]

```

**To run the code**

1. Give proper path to the video/image file
2. Download the weight file "yolov3.weights" for YOLOv3 and set proper path in yolo.py
2. Run `bash run.sh`

**Files**

* run.sh -- to run the code
* yolo.py -- performs object detection 
* yolov3.cfg -- configuration file
* coco.names -- object classes available
