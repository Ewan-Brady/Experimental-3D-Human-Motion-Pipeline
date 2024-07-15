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


def pose_addition(skeleton_file, depth_directory, point_cloud video_name):
    skeleton_frames = np.load(skeleton_file)
    num_frames = skeleton_frames.shape[0]
    
    #NOTE: For now code assuming there is only 1 individual in the image, but what happens if there is multiple?
    processed_skeleton_frames = []
    for i in range(num_frames): #iterate over each frame
        #First construct a mini-pointcloud for just the pose points of the frame. 

        #Copypasted from pointcloud functions, use this to get depth for frame
        depth_frame = (depth_directory+"/"+video_name+"_frame" + str(i) + ".npy")

        depth_frame = np.load(depth_frame)
        depth_frame = np.transpose(depth_frame, (1, 2, 0))
        depth_frame = cv2.resize(depth_frame, (240, 320))
        depth_frame = np.expand_dims(depth_frame, axis = 2)
        depth_frame = np.max(depth_frame)-depth_frame

        depth_frame = depth_frame * depth_multiplier

        limb_points = []
        for j in skeleton_frames[i]:
            xloc = round(j[0])
            yloc = round(j[1])
            depth = depth_frame[xloc][yloc] #get the depth at that point
            limb_points.append(np.array([xloc,yloc,depth]))
        limb_points = np.stack(limb_points) #Stack limb points into a mini pointcloud.
        limb_points = cloud_FOV_spread(limb_points, FOV, FOV, 320, 240) #Do FOV spread on limb points
        
        limb_angles = calculate_body_angles(limb_points)
        
        processed_skeleton_frames = np.concatenate(limb_points,limb_angles)

        #Here we need to calculate all of the angles on the body from 
        
    return processed_skeleton_frames, point_cloud

"""
Given 3d keypoints, return robot limb orientations

Keypoints are as follows
First point is somewhere like the nose/mouth
Second point is the "left eye" (on right side of face if facing viewer in image)
Third point is the "right eye"
Fourth point is the "left ear"
Fifth point is the "right ear"
Sixth point "left shoulder"
Seventh point "right shoulder"
Eighth point "left elbow"
Ninth point: "right elbow"
Tenth point: "left hand:"
Eleventh point: "right hand:" 
Twelth point: "left hip"
Thirteenth point: "right hip"
Fourtheenth point: "left hip"
Fiftheetnh point: "right hip"
Sixteenth point: "left foot"
Seventeetnh point: "right foot"


Output is as follows:
Torso-Head connection
    -torso_x
    -torso_y
    -torso_z
Abdomen-torso connection
    -abdomen_z
    -abdomen_y
    -abdomen_x
Right_Hip-Right_Thigh connection
    -right_hip_x
    -right_hip_z
    -right_hip_y
Right_Thigh-Right_Calf connection
    -right_knee
Left_Hip-Left_Thigh connection
    -left_hip_x
    -left_hip_z
    -left_hip_y
Left_Thigh-Left_Calf connection
    -left_knee
Torso-Right_Bicep connection
    -right_shoulder1 (up/down, doing chicken)
    -right_shoulder2 (forward/back, swinging arms)
Right_Bicep-Right_Forearm connection
    -right_elbow
Torso-Left_Bicep connection
    -left_shoulder1 (up/down, doing chicken)
    -left_shoulder2 (forward/back, swinging arms)
Left_Bicep-Left_Forearm connection
    -left_elbow
    
Can see mujoco model to see oreintations in action.
MAKE SURE TO CHECK THAT "LEFT" vs "RIGHT" is consistent.

My current idea is we can do the following to get the orientations from the points.:
Can use ears, eyes, and the mouth/nose to approximate the orientation and location of the head.
Can use discrepancy between line from shoulder-to-shoulder and line from hip-to-hip to approximate torso-abdomen orientation.
Can use discrepancy between shoulder-to-shoulder midpoint and estimated head location to get head-torso orientation.
Can use elbow point, shoulder-shoulder line, and position of head (to indicate "up") to get Bicep-Torso orientation.
Can use the elbow point, shoulder point, and hand point to get elbow orientation.
Can use hip-to-hip line, knee position, and head position (to indicate "up") to get Hip-Thigh orientation.
Can use hip point, knee point, and foot point to get the knee orientation.
"""
def calculate_body_angles(body_points):
    print("In progress")

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
"""