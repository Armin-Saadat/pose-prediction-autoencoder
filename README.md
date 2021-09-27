# Towards Human Pose Prediction using the Encoder-Decoder LSTM (SoMoF)

## _Absract_:

_Human pose prediction is defined as predicting the hu-man  keypoints  locations  for  future  frames,  given  the  ob-served ones for past frames.  It has numerous applicationsin various fields like autonomous driving.  This task can beseen as a fine-grained task while human bounding box pre-diction  deals  with  more  coarse-grained  information.   Theformer has been investigated less and here, we modify oneof the previously-used architectures of bounding box pre-diction to do a harder task of pose prediction in the SoMoFchallenge.   The results show the effectiveness of the pro-posed method in evaluation metrics.

## Introduction:
This is the official code for the Abstract ["Towards Human Pose Prediction using the Encoder-Decoder LSTM"](link), accepted and published in ["ICCVW 2021"](https://somof.stanford.edu/workshops/iccv21)
You can find report in [here]() 

## Contents
------------
  * [Repository Structure](#repository-structure)
  * [Proposed Method](#proposed-method-DeRPoF)
  * [Results](#results)
  * [Installation](#installation)
  * [Dataset](#dataset)
  * [Training/Testing/Predicting](#training-testing)
  * [Tested Environments](#tested-environments)

```
posepred
├── dataloader
│   ├── de_global_dataloader.py         -- dataloader for global part of disentangling model
|   ├── de_local_dataloader.py          -- dataloader for local part of disentangling model
|   ├── de_predict_dataloader.py        -- dataloader for predicting
|   └── lstm_vel_dataloader.py          -- dataloader fo lstm_vel model
├── Posetrack
|   ├── posetrack_test_in.json          -- posetrack dataset that is prepared by SoMoF for the challenge
|   └── ...                             -- argument handler for other modules
├── train_scripts
|   ├── de_global_posetrack.py          -- training global joint for disentangling model on posetrack
|   ├── de_local_posetrack.py           -- training local joints (all except neck-joint) for disentangling model on posetrack
|   ├── lstm_vel_posetrack.py           -- training lstm_vel (proposed method) on posetrack
|   └── lstm_vel_3dpw.py                -- training lstm_vel (proposed method) on 3dpw
├── predict
|   ├── lstm_vel_posetrack.py           -- Predict upon test data in /Posetrack on lstm_vel (main proposed) model
|   ├── lstm+vel_3dpw.py                -- Predict upon test data on lstm_vel (main proposed) model
│   ├── disentangling_posetrack.py      -- Predict upon test data in /Posetrack on disentangling model
|   ├── last_observed_pose.py           -- defining a baseling with repeating last observed pose for all prediction frames
|   ├── last_observed_speed.py          -- defining a baseling with repeating last observed speed for all prediction frames
|   └── ...
├── models
│   ├── decoder.py                      -- base code for decoder
|   ├── encoder.py                      -- base code for encoder
│   ├── disentangle1.py                 -- first disentangle model using pv_lstm structure
|   ├── lstm_vel_3dpw.py                -- Proposed model on 3dpw
|   ├── de_global_posetrack             -- Disentangling model for global joint on poestrack
|   └── de_local_posetrack.py           -- Disentangling model for local joint on poestrack
├── preprocessed_csvs
|   ├── 3dpw_train.csv                  -- preprocessed training set for 3dpw
|   ├── 3dpw_valid.csv                  -- preprocessed validation set for 3dpw
|   ├── posetrack_train.csv             -- preprocessed training set for posetrack
|   └── posetrack_valid.csv             -- preprocessed validation set for posetrack
├── utils
|   ├── visualizer.py                   -- visualizing predicted poses
|   ├── metrics.py                      -- available metrics
|   ├── option.py                       -- parse arguments handler
|   ├── save_load.py                    -- base code for saving and loading models
|   └── others.py                       -- other useful utils

```

## Proposed method LSTMV_LAST
-------------
We decouple the pose forecasting into a global trajectory forecasting and a local pose forecasting as shown below:
![Our proposed method](images/network.png)


## Results

We show the observed (left) and the predicted (right) poses for two different scenarios. The rows correspond to DeRPoF w/o early stop and w/o Decoupling from top to bottom. Only the pose of every other frame is shown. 
![a](figures/fig4--a.png)
![b](figures/fig4--b.png)

## Installation:
------------
Start by cloning this repositiory:
```
git clone https://github.com/Armin-Saadat/SoMoF.git
cd decoupled-pose-prediction
```
Create a virtual environment:
```
virtualenv myenv
source myenv/bin/activate
```
And install the dependencies:
```
pip install -r requirements.txt
```

## Dataset:
  
  * We use the preprocessed posetrack and 3dpw datasets in [SoMoF](https://somof.stanford.edu/dataset) challenge. For easy usage,these datasets are preprocessed. The clean version of dataset is available at /preprocess_csvs. 
  
## Training/Validating/Predicting:
In order to train the model for posetrack:

```
cd train_scripts
python3 -m lstmvel_posetrack
```

To train the model on 3DPW:

```
cd train_scripts
python3 -m lstmvel_3dpw
```
Model also is validating each epoch on training section.

The output will be the vim and vam values also you can visualize your outpurs using utils/vis.py .

Test the trained network on posetrack:
```
cd predict
python lstmvel_posetrack.py --load_ckpt=<path_to_saved_snapshot.pth>
```

Test and predict the trained network on 3dpw:
```
cd predict
python lstmvel_3dpw.py --load_ckpt=<path_to_saved_snapshot.pth>
```
where other options are similar to the training. 

We also have implemented many other models that you can see in models/ directory. If you want to run those, you have to repeat aforementioned procedure for those models. 

## Tested Environments:
------------
  * Ubuntu 20.04, CUDA 10.1
 
