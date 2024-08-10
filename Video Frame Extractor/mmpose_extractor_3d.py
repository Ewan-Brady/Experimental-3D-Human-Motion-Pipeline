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
from mmpose.apis import MMPoseInferencer
import os
import numpy as np
import cv2

confidence_threshhold = 0.8 #discard results below this level of confidence

#Old args
mmpose_path = "/mnt/c/AI_model/3DMovementModel/Video Pose Estimator/mmpose/"

config = mmpose_path+"td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
checkpoint = mmpose_path+"td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"
device = "cuda:0"

#img = mmpose_path+"tests/data/coco/000000000785.jpg"

inputDirectory = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted Images" #The absolute directory where the image video frames are stored
outputDirectory = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Pose Estimations" #The absolute directory where the skeleton estimations should be outputted.

def extract_2d_frames(model, frames_folder_2d):
    to_return_to = os.getcwd() #for later use
    
    os.chdir(frames_folder_2d) #go into target directory
    images = os.listdir()
            
    keypoint_frames = []
            
    frames_visibility = []
    non_visible_count = 0
    
    faulty_frames = set()
    i = 0
    strikes2 = 0
    for img in images:
        # inference a single image
        batch_results = inference_topdown(model, img)
        results = merge_data_samples(batch_results).pred_instances
        
        strikes = 0
        
        frame_visible = True
        for i2 in range(5): #Check first 5 keypoints are in bounds.
            if(results.keypoints_visible[0][i2] < confidence_threshhold): #Check for likely not visible head points.
                strikes = strikes+1
                if(strikes > 4):
                    frame_visible = False
        if(frame_visible == False):
            strikes2 = strikes2+1
        else:
            strikes2 = 0
            
        #frames_visibility.append(frame_visible)
        keypoint_frames.append(results.keypoints[0]) #Add frame of data
        
        if(strikes2 > 4): #This system is in place to avoid one-off below confidence frames screwing things up.
            faulty_frames.add(i-4)
            faulty_frames.add(i-3)
            faulty_frames.add(i-2)
            faulty_frames.add(i-1)
            faulty_frames.add(i)
        i = i + 1
     
    for i3 in faulty_frames: #Replaces faulty frames with -1 to indicate faulty. 
        keypoint_frames[i3] = np.full(keypoint_frames[i3].shape,-1)
    """
    This code discards videos where the character is not meaningfully in frame.
    However deciding against using this code, that filtering can be done at the final stage where
    we can split it into multiple usable frames (allows for handling of movie cuts to different angles by
    separating them into two separate pieces of training data)
    
    middle_start_end = 0 #starts at 0, then goes to 1, then goes to 2. Any deviation from this means discard
    #because there is a pocket where the head is not visible
    #This filters so only videos where there is a solid middle section of the head being visible are kept.
    last_truth = False
    for i in frames_visibility:
        if i != last_truth: #change in truth value
            middle_start_end = middle_start_end+1 #move to next stage
            last_truth = i #new truth value
    if not ((middle_start_end == 2 and last_truth == False) or (middle_start_end == 1 and last_truth == True)):
        print(frames_visibility)
        return None
    
    """
        
    keypoint_frames = np.stack(keypoint_frames)
    os.chdir(to_return_to) #return to main directory.
    
    return keypoint_frames


def extract_3d_frames(inferencer, img_path):
    to_return_to = os.getcwd() #for later use
    
    os.chdir(img_path) #go into target directory
    images = os.listdir()
            
    keypoint_frames = []
            
    faulty_frames = set()
    strikes = 0
    i = 0
    for img in images:
        image = cv2.imread(img)
        image = cv2.resize(image, (640, 480))

        # inference a single image
        result_generator = inferencer(image, show=False)
        result = next(result_generator)
        if (len(result["predictions"][0]) > 1): #Check for a number of people not equal to one.
            strikes = strikes+1 #Sometimes it hallucinates an extra figure, use a strike system to weed out these cases.
            if(strikes > 2):
                #This and previous striked frames are flawed because they contain multiple individuals.
                faulty_frames.add(i-2)
                faulty_frames.add(i-1)
                faulty_frames.add(i)
                #return None
        else:
            strikes = 0
        
        keypoint_frames.append(np.array(result["predictions"][0][0]["keypoints"])) #Add frame of data
        i = i + 1

    for i2 in faulty_frames: #Replaces faulty frames with -1 to indicate faulty. 
        keypoint_frames[i2] = np.full(keypoint_frames[i2].shape,-1)
        
    keypoint_frames = np.stack(keypoint_frames)
    os.chdir(to_return_to) #return to main directory.

    return keypoint_frames


def main():
    # build the model from a config file and a checkpoint file
    cfg_options = None

    os.chdir(mmpose_path)
    inferencer = MMPoseInferencer(pose3d="human3d")

    model = init_model(
        config,
        checkpoint,
        device=device,
        cfg_options=cfg_options)
    
    os.chdir(inputDirectory)
    actions = os.listdir()
    os.chdir(outputDirectory) #now, create corresponding output directories for each action
    for action in actions: 
        action_path_2d = outputDirectory + "/2D/" + action
        action_path_3d = outputDirectory + "/3D/" + action
        if(not os.path.exists(action_path_2d)):
            os.makedirs(action_path_2d) #Create action output directory if it does not exist
        if(not os.path.exists(action_path_3d)):
            os.makedirs(action_path_3d) #Create action output directory if it does not exist

    for action in actions:
        action_directory = inputDirectory + "/" + action
        os.chdir(action_directory)
        frame_folders = os.listdir() #get a list of the input image folders.
        
        skip_occured = False
        skips = 0
        for folder in frame_folders:
            frame_folder = action_directory+"/"+folder
            target_location_3D = outputDirectory + "/3D/" + action + "/"  + folder
            target_location_2D = outputDirectory + "/2D/" + action + "/"  + folder

            if(os.path.exists((target_location_3D+".npy")) and os.path.exists((target_location_2D+".npy"))): #Skips finished files to resume.
                skip_occured = True
                skips = skips + 1
                continue
            if(skip_occured):
                print("Skipped " + str(skips) + " times to " + folder)
                skips = 0
                skip_occured = False

            keypoint_frames_3D = extract_3d_frames(inferencer, frame_folder)

            keypoint_frames_2D = extract_2d_frames(model, frame_folder)
            
            np.save(target_location_2D, keypoint_frames_2D)
            np.save(target_location_3D, keypoint_frames_3D)

            
        print(action + " done...")
        #with open(target_location, "w") as imagefile:
        #    imagefile.write(str(keypoint_frames)) #Writes skeletondata frames to file, pretty simple format but can be decoded so it works.
            

if __name__ == '__main__':
    main()

