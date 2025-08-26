# MoNet

Code for our paper:  
**MoNet: Decomposing Motion Trajectories with a Hybrid Network for Human Motion Prediction**  
Shuaijin Wan, Huaijiang Sun

## Recommand Dependencies
* Cuda 11.7
* Python 3.8
* Pytorch 1.13.1

## Data preparation
### Human3.6M
Download Human3.6M dataset from its [website](http://vision.imar.ro/human3.6m/description.php) and put the files into "data" with the dataset folder named "h3.6m".

### CMUMocap
Download CMUMocap dataset from its [website](http://mocap.cs.cmu.edu) and put the files into "data" with the dataset folder named "CMUMocap".
```
/data
├── h36m
    ├── dataset
        ├── S1
        ├── S5
        ├── ...
├── CMU
    ├── test
    ├── train        
```

## Train
### Human3.6M
To train a motion prediction model, run
```
python main.py
```

The experimental results will be saved in exp/ .

### CMUMocap
To train a motion prediction model, run
```
python main_cmu.py
```

The experimental results will be saved in exp/ .

### Evaluate
### Human3.6M
To evaluate a motion prediction model, run
```
python main.py --mode eval 
```

