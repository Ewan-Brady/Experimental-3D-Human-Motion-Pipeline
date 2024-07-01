#An alterted version of the mmpose installation demo
#Conda environment VideoPoseEstimator
#Relies on mmpose being installed, mmpose_path variable points to it.


import logging
from argparse import ArgumentParser

from mmcv.image import imread
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
import os
import numpy as np

#Old args
mmpose_path = "/mnt/c/AI_model/3DMovementModel/Video Pose Estimator/mmpose/"

config = mmpose_path+"td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
checkpoint = mmpose_path+"td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"
device = "cuda:0"

#img = mmpose_path+"tests/data/coco/000000000785.jpg"

inputDirectory = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted Images" #The absolute directory where the image video frames are stored
outputDirectory = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Pose Estimations" #The absolute directory where the skeleton estimations should be outputted.


def main():
    # build the model from a config file and a checkpoint file
    cfg_options = None

    model = init_model(
        config,
        checkpoint,
        device=device,
        cfg_options=cfg_options)
    
    os.chdir(inputDirectory)
    actions = os.listdir()
    os.chdir(outputDirectory) #now, create corresponding output directories for each action
    for action in actions: 
        action_path = outputDirectory + "/" + action
        if(not os.path.exists(action_path)):
            os.makedirs(action_path) #Create action output directory if it does not exist

    for action in actions:
        action_directory = inputDirectory + "/" + action
        os.chdir(action_directory)
        frame_folders = os.listdir() #get a list of the input image folders.
        
        for folder in frame_folders:
            os.chdir(action_directory+"/"+folder)
            images = os.listdir()
            
            keypoint_frames = []
            
            for img in images:
                # inference a single image
                batch_results = inference_topdown(model, img)
                results = merge_data_samples(batch_results)
    
                if(len(results.pred_instances.keypoints) > 1):
                    raise Exception("Multiple keypoints in " + img)
                
                keypoint_frames.append(results.pred_instances.keypoints[0]) #Add frame of data
            
            keypoint_frames = np.concatenate(keypoint_frames)

            target_location = outputDirectory + "/" + action + "/"  + folder
            
            np.save(target_location, keypoint_frames)
            
        print(action + " done...")
        #with open(target_location, "w") as imagefile:
        #    imagefile.write(str(keypoint_frames)) #Writes skeletondata frames to file, pretty simple format but can be decoded so it works.
            


    """
    # inference a single image
    batch_results = inference_topdown(model, img)
    results = merge_data_samples(batch_results)
    
    print(results.pred_instances.keypoints) #THIS LINE HERE WAS ADDED BY ME, it prints the keypoints, though for some reason there is an empty dimensions beforehand in the array.
    #Each keypoint is stored as a pair, the pair is a 2 member list
    print(results.pred_instances.keypoints[0]) #This compensates for the extra dimension
    #I suspect the extra dimension implies it is possible to do batches of images, where each first dimension layer is one of the batch members.
    """

if __name__ == '__main__':
    main()

