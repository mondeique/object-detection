# object-detection
Object-Detection API using MSCOCO dataset from Tensorflow & customized object-detection

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
> 편의를 위해 conda virtualenv는 생략했다. (mondeique)
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
### How to test with MSCOCO dataset
```
$ CUDA_VISIBLE_DEVICES=0 python object_detection_run.py
```
### How to install pycocotools
- cocoapi를 local에 clone한다.
```
$ git clone https://github.com/philferriere/cocoapi.git
```
- setup 이동경로로 가서 pycocotools를 설치한다.
```
$ python setup.py install 
```
## How to train with customized dataset

1. split labels for training & evaluation with split_labels.ipynb
2. download image from s3 with [data-explorer](https://github.com/mondeique/data-explorer)
3. python generate_tfrecord.py 
```
$ python generate_tfrecord.py --csv_input=data/training_bag_csv --output_path=data/train.record --image_dir=images/
$ python generate_tfrecord.py --csv_input=data/test_bag_csv --output_path=data/test.record --image_dir=images/
```

4. change pipeline.config with selected network
> 원하는 network config 변경 가능 : hyperparameter-tuning (ex : batch_size, learning rate ...)
5. create object-detection.pbtxt
> item - id - class
6. python model_main.py
```
$ python model_main.py --pipeline_config_path=training/pipeline.config --model_dir=training/ \
    --num_train_steps=${NUM_TRAIN_STEPS} --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
```
## Data Structure
```
  Project
  |--- data
  |    |--- test_bag_csv
  |    |--- training_bag_csv
  |    |--- train.record
  |    |--- test.record
  |--- images (for training/evaluation)
       |--- image1.jpg
       |--- image2.jpg
       |---
       ...
  |--- test_images
       |--- test_image1.jpg
       |--- test_image2.jpg
       |---
       ...
  |
  |--- my_network_output
       |--- saved_model
       |--- pipeline.config
       |--- frozen_inference_graph.pb
       |--- model.ckpt
       |--- checkpoint
  |--- label_map
       |--- mscoco_label_map.pbtxt (for minimal working)
  |--- test_result
       |--- my_network_for_customized
       |--- faster_rcnn_resnet101 (for minimal working)
       |--- ssd_mobilenet_v1 (for minimal working)
       ...
  |--- utils (for import label_map_util)    
  |--- training
       |        
       |--- model.ckpt 
       |--- object-detection.pbtxt
       |--- pipeline.config
  |--- generate_tfrecord.py
  |--- export_inference_graph.py
  |--- model_main.py
  |--- object_detection_run.py (for test)
  |--- split_labels.ipynb   
  ```
## Test Result
### Minimal Working Test 
> [More Test Result](https://github.com/mondeique/object-detection/tree/master/test_result)
#### Faster R-CNN resnet101
![result_img](./test_result/faster_rcnn_resnet101/result_img2.jpg)
![result_img](./test_result/faster_rcnn_resnet101/result_img3.jpg)
#### ssd inception v2
![result_img1](./test_result/ssd_inception_v2/result_img2.jpg)
![result_img1](./test_result/ssd_inception_v2/result_img3.jpg)
#### ssd mobilenet v1
![result_img1](./test_result/ssd_mobilenet_v1/result_img2.jpg)
![result_img1](./test_result/ssd_mobilenet_v1/result_img3.jpg)
### Customized Test
- tensorboard 로 현재 학습되고 있는 images와 loss를 확인할 수 있다. 
```
$ tensorboard --logdir='training'
```
- inference graph를 뽑아 test를 위한 준비를 한다.
```
$ python export_inference_graph.py --input_type=image_tensor --pipeline_config_path=pipeline.config \
--trained_checkpoint_prefix=training/model.ckpt-** --output_directory=output/
```
```
$ CUDA_VISIBLE_DEVICES=0 python object_detection_run.py
```
#### ssd mobilenet v1 for 8549장 handbag dataset (handle 포함)
![result_img1](./test_result/ssd_mobilenet_v1_output/result_img2.jpg)
![result_img1](./test_result/ssd_mobilenet_v1_output/result_img3.jpg)
- __tfrecord 변환 과정에서 생긴 error였다.__
- Batch Size : 24 / number of steps : 100000
> 72h : about 40000 steps
#### ssd mobilenet v1 for 1000장 handbag dataset (handle 포함)
![result_img1](./test_result/ssd_mobilenet_v1_output_eren/result_img2.jpg)
![result_img1](./test_result/ssd_mobilenet_v1_output_eren/result_img3.jpg)
- error 해결하고 나온 첫 성공결과
- __손잡이 제외한 부분을 뽑아내기 위해서는 다시 cropped 과정을 거쳐야 한다!__
- Batch Size : 16 / number of steps : 40000
> about 36h : 40000 steps 
#### ssd mobilenet v1 for 1000장 handbag dataset (handle 제외)
![result_img1](./test_result/ssd_mobilenet_v1_1000_16_30000_no_handle/loss_16_30000.png)
- 수렴이 되는 느낌은 나지만 안정적으로 수렴이 되진 않는다.<br></br>
![result_img1](./test_result/ssd_mobilenet_v1_1000_16_30000_no_handle/result_img2.jpg)
![result_img1](./test_result/ssd_mobilenet_v1_1000_16_30000_no_handle/result_img3.jpg)
- 사람이 들고 있는 사진이 없었기 때문에 그런 경우는 제대로 가방을 찾지 못하였음.
- 손잡이 제외 사람이 들고 있는 사진까지 포함하면 더 이상 건드리지 않아도 됨!
- Batch size : 16 / number of steps : 30000
> about 30h : 30000 steps
#### ssd mobilenet v1 for 1500장 handbag dataset (handle 제외)
![result_img1](./test_result/ssd_mobilenet_v1_1500_16_15000_people_no_handle/result_img7.jpg)
![result_img1](./test_result/ssd_mobilenet_v1_1500_16_15000_people_no_handle/result_img10.jpg)
- 사람이 들고 있는 사진까지 포함하였기 때문에 잘 찾는 것을 확인함.
![result_img1](./test_result/ssd_mobilenet_v1_1500_16_15000_people_no_handle/result_img13.jpg)
- 다중 가방 포함된 Data 부족으로 인한 handbag detection 불가 : 추가 발전 필요
- Batch size : 16 / number of steps : 15000
## TODO 

- [X] dev env setting (ubuntu 18.04) : decide on 2019.10.09
- [X] GPU testing
- [X] protobuf testing : 2019.10.10
- [X] Object-Detection API test : 2019.10.10
- [X] split training / evaluation csv : 2019.10.11
- [X] data-explorer from each csv : 2019.10.11
- [X] generate_tfrecord.py : 2019.10.11
- [X] pipeline.config to ssd mobilenet으로 변경 : 2019.10.12
- [X] pbtxt 생성 : 2019.10.12
- [X] model_main.py : training : 2019.10.12-2019.10.14
- [X] export_inference_graph.py : 2019.10.15
- [X] test : object_detection_run.py file 수정을 통한 test 과정 : 2019.10.15
- [X] error analysis : bounding box feature xmin/xmax/ymin/ymax : 2019.10.16
- [X] cropping detection image code : object detection boxes 정보 (ltrb) 를 이용하여 image cropping

### Reference
> [tensorflow official object-detection models](https://github.com/tensorflow/models/tree/master/research/object_detection)
