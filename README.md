# libppe

# setup environments
```
1. create python(python3) virtual environments
    $ python -m venv venv

2. install python packages with pip(pip3)
    $ pip install -r requirements.txt

```

# Forktip detection model training with pytorch
```
1. move the current directory
    $ cd ./pyppe/fork_dataset/yolo

2. download the yolov5 model from git
    $ git clone https://github.com/ultralytics/yolov5.git

3. after then, install required packages to train our model and copy the dataset files
    $ cd yolov5
    $ mkdir dataset
    $ cp ../../../dataset/* ./dataset
    $ python train.py --img 1280 --batch 16 --epochs 20 --data ./dataset/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name effector

4. find the model file(*.pt) in runs/train/effector directory

```
