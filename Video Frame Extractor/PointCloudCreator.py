#Conda environment VideoFrameExtractor
from os import replace
import cv2
import numpy as np
import math
import os

depth_multiplier = 2 #Can be any number, set it to two as I think it makes pointclouds look better/more accurate
FOV = 53 * math.pi/180
frame_dividor = 3
#depth_threshhold_fraction = 2/3

def convert_directory(image_directory, depth_directory, video_name):
    to_return = []
    num_frames = len(os.listdir(image_directory))
    for i in range(num_frames):
        image_frame = (image_directory+"/"+video_name+"_frame" + str(i) + ".jpg")
        depth_frame = (depth_directory+"/"+video_name+"_frame" + str(i) + ".npy")
        frame = convert_to_pointcloud(image_frame,depth_frame)
        to_return.append(frame)
    to_return = np.stack(to_return)
    return to_return

def convert_to_pointcloud(image, depth):
    

    image = cv2.imread(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = cv2.resize(image, (240, 320))
    image = image * 255
    #print(image.shape)


    depth = np.load(depth)
    depth = np.transpose(depth, (1, 2, 0))
    depth = cv2.resize(depth, (240, 320))
    depth = np.expand_dims(depth, axis = 2)
    depth = np.max(depth)-depth
    
    #depth_threshhold = np.max(depth)*depth_threshhold_fraction
    
    depth = depth * depth_multiplier
    #print(depth.shape)

    cloud_preprocessed = np.concatenate([depth, image], axis = 2)
    cloud_preprocessed = np.transpose(cloud_preprocessed, (1,0,2))
    #cloud_preprocessed = np.flip(cloud_preprocessed, axis = 0)
    cloud_preprocessed = np.flip(cloud_preprocessed, axis = 1)
    #print(cloud_preprocessed.shape)
    #print(cloud_preprocessed.shape)
    
    cloud = []
    for i in range(len(cloud_preprocessed)):
        for j in range(len(cloud_preprocessed[i])):
            if(i%frame_dividor==0 and j%frame_dividor==0):
                #if(cloud_preprocessed[i][j][0] < depth_threshhold):
                xy_point = np.array([i, j])
                cloud.append(np.concatenate([xy_point, cloud_preprocessed[i][j]]))
    cloud = np.stack(cloud)
    
    cloud = cloud_FOV_spread(cloud, FOV, FOV, 320, 240)
    return cloud

def cloud_FOV_spread(array, angle_horizontal, angle_vertical, width, height):
    #At the edge it should be at full angle, while at the middle there should be no rotation.
    #Note that they do not start at a 0 degree angle, and we do not want them to rotate extra.
    width_middle = width/2
    height_middle = height/2
    POVDepth = width_middle/math.tan(angle_horizontal)#*width_middle
    ratio1 = 0
    ratio2 = 0
    for point in array:
        magnitude = np.sqrt(np.sum(np.square(np.array([(point[0]-height_middle), (point[1]-width_middle), point[2]]))))
        direction_vector = np.array([point[0],point[1],0])-np.array([height_middle,width_middle,POVDepth])
        direction_vector = direction_vector/np.sqrt(np.sum(np.square(direction_vector)))
        #print(point[0])

        direction_vector = direction_vector*magnitude#(point[2]+POVDepth)#magnitude #Now this should be the new point

        point[0] = direction_vector[0]
        point[1] = direction_vector[1]
        #point[2] = direction_vector[2]

    return array

def numpy_vid_to_text(array):
    to_return = ""
    for i in array:
        frame_string = numpy_to_text(i)
        to_return = to_return+frame_string + "\n/\n"
    return to_return

def numpy_to_text(array):
    toreturn = ""
    points = []
    for i in array:
        #replacement = str(i)
        #replacement = replacement.replace("[", "")
        #replacement = replacement.replace("]", "")
        #replacement.replace(" ", ",")
        #replacement = replacement.replace(". ", ",")
        #replacement = replacement.replace(".", "")
        #print(replacement)
        replacement = ""
        nums = []
        for j in i:
            nums.append(str(j))
        replacement = ",".join(nums)
        points.append(replacement)
    toreturn = "\n".join(points)
    return toreturn
"""
Demonstration code:
image_directory = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted Images/run/50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi"
depth_directory = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted Depths/run/50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi"
video = "50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi"

array = convert_directory(image_directory, depth_directory,video)
stringmade = numpy_vid_to_text(array)

with open("test.txt", "w") as text_file:
    text_file.write(stringmade)


Copied the below codee from Video_Frame_Extractor.py and modified it. 

The below iterator is made to extract from the HMDB51 dataset's directory structre. 
The frame extraction function however works for whatever, it just spits out its output images
into whatever is set as the current directory for the program, and you can feed an absolute path
into the function as input. 
"""


inputDirectory = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted Images" #The absolute directory where the input video dataset is stored.
depthDirectory = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted Depths"
outputDirectory = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted Point Clouds" #The absolute directory where you want it to spit out the images.

os.chdir(inputDirectory) #First, go to the input directory and get its members
actions = os.listdir() #got its members (and in HMDB51 also corresponds to main action)


os.chdir(outputDirectory) #now, create corresponding output directories for each action
for action in actions: 
    action_path = outputDirectory + "/" + action
    if(not os.path.exists(action_path)):
        os.makedirs(action_path) #Create action output directory if it does not exist


for action in actions: #now that we have created the nessecary directories, we can extract the images into them
    os.chdir(inputDirectory + "/" + action)
    videos = os.listdir() #get a list of the input videos.
    
    
    action_output_directory = (outputDirectory + "/" + action)
    os.chdir(action_output_directory) #return to output directory for action

    for video in videos:
        depth_location = depthDirectory +"/" + action + "/" + video
        image_location = inputDirectory +"/" + action + "/" + video
        output = convert_directory(image_location,depth_location,video)    
        target_location = action_output_directory + "/" + video + "_pointcloud"

        np.save(target_location,output)
        
    print(action + " done...")

print("image extraction complete!")
