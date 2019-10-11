# object-detection
Object-Detection API using MSCOCO dataset from Tensorflow

## How to install 
[Object-Detection API Install Guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
```
$ sudo pip install -r requirements.txt
```
### protocol buffer test
```
$ conda activate mondeique
(mondeique) $ sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
(mondeique) $ sudo pip install pillow
(mondeique) $ sudo pip install jupyter
(mondeique) $ sudo pip install matplotlib 
```
- 편의를 위해 conda virtualenv는 생략했다. (mondeique)
- Object Detection API는 protocol buffer를 이용한다. (실행시 아래 test를 계속 해줘야 한다)
```
$ git clone http://github.com/tensorflow/models
$ cd models/research

$ protoc object_detection/protos/*.proto --python_out=.
```
- 환경변수를 설정해준다.
```
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
- 설치가 제대로 되었는지 확인한다. 
```
$ python object_detection/builders/model_builder_test.py
```
## How to run 
```
$ CUDA_VISIBLE_DEVICES=0 python object_detection_run.py
```
## Test Result
### Faster R-CNN resnet101
![result_img](./test_result/faster_rcnn_resnet101/result_img2.jpg)
![result_img](./test_result/faster_rcnn_resnet101/result_img3.jpg)
### Faster R-CNN resnet50
![result_img](./test_result/faster_rcnn_resnet50/result_img2.jpg)
![result_img](./test_result/faster_rcnn_resnet50/result_img3.jpg)
### Faster R-CNN inception v2
![result_img](./test_result/faster_rcnn_inception_v2/result_img2.jpg)
![result_img](./test_result/faster_rcnn_inception_v2/result_img3.jpg)
### Mask R-CNN inception v2
![result_img](./test_result/mask_rcnn_inception_v2/result_img2.jpg)
![result_img](./test_result/mask_rcnn_inception_v2/result_img3.jpg)
### ssd inception v2
![result_img1](./test_result/ssd_inception_v2/result_img2.jpg)
![result_img1](./test_result/ssd_inception_v2/result_img3.jpg)
### ssd resnet50 v1 fpn
![result_img1](./test_result/ssd_resnet50_v1_fpn/result_img2.jpg)
![result_img1](./test_result/ssd_resnet50_v1_fpn/result_img3.jpg)
### ssd mobilenet v1
![result_img1](./test_result/ssd_mobilenet_v1/result_img2.jpg)
![result_img1](./test_result/ssd_mobilenet_v1/result_img3.jpg)
### ssd mobilenet v2
![result_img1](./test_result/ssd_mobilenet_v2/result_img2.jpg)
![result_img1](./test_result/ssd_mobilenet_v2/result_img3.jpg)
## TODO 

- [X] dev env setting (ubuntu 18.04) : decide on 2019.10.09
- [X] GPU testing
- [X] protobuf testing : 2019.10.10
- [X] Object-Detection API test : 2019.10.10
- [ ] fine-tuning coding
- [ ] error analysis : 아예 못 잡는 경우 / 작게 잡는 경우

