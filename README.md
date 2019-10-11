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
## Data Structure
__TO BE ADDED....__
## Test Result
### Minimal Working Test
#### Faster R-CNN resnet101
![result_img](./test_result/faster_rcnn_resnet101/result_img2.jpg)
![result_img](./test_result/faster_rcnn_resnet101/result_img3.jpg)
#### Faster R-CNN inception v2
![result_img](./test_result/faster_rcnn_inception_v2/result_img2.jpg)
![result_img](./test_result/faster_rcnn_inception_v2/result_img3.jpg)
#### Mask R-CNN inception v2
![result_img](./test_result/mask_rcnn_inception_v2/result_img2.jpg)
![result_img](./test_result/mask_rcnn_inception_v2/result_img3.jpg)
#### ssd inception v2
![result_img1](./test_result/ssd_inception_v2/result_img2.jpg)
![result_img1](./test_result/ssd_inception_v2/result_img3.jpg)
#### ssd resnet50 v1 fpn
![result_img1](./test_result/ssd_resnet50_v1_fpn/result_img2.jpg)
![result_img1](./test_result/ssd_resnet50_v1_fpn/result_img3.jpg)
#### ssd mobilenet v1
![result_img1](./test_result/ssd_mobilenet_v1/result_img2.jpg)
![result_img1](./test_result/ssd_mobilenet_v1/result_img3.jpg)
#### ssd mobilenet v2
![result_img1](./test_result/ssd_mobilenet_v2/result_img2.jpg)
![result_img1](./test_result/ssd_mobilenet_v2/result_img3.jpg)
## TODO 

- [X] dev env setting (ubuntu 18.04) : decide on 2019.10.09
- [X] GPU testing
- [X] protobuf testing : 2019.10.10
- [X] Object-Detection API test : 2019.10.10
- [ ] split training / evaluation csv : 2019.10.11
- [ ] data-explorer from each csv : 2019.10.11
- [ ] generate_tfrecord.py : 2019.10.11
- [ ] pipeline.config to ssd mobilenet으로 변경 : 2019.10.12
- [ ] pbtxt 생성 : 2019.10.12
- [ ] model_main.py : training : 2019.10.12-2019.10.14
- [ ] error analysis 

