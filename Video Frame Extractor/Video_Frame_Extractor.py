#Used this: https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/
#Conda environment VideoFrameExtractor

import cv2 

import os
  
# Function to extract frames 
def FrameCapture(path, video_name): 
  
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
    
    error_frames_count = 0
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
        try:
            # Saves the frames with frame-count 
            cv2.imwrite(video_name+("_frame%d.jpg" % count), image) 
  
            count += 1
        except:
            error_frames_count += 1
            
    if error_frames_count > 1:
        print(video_name + " had an unexpected number of error frames, " + str(error_frames_count))
        
"""
The below iterator is made to extract from the HMDB51 dataset's directory structre. 
The frame extraction function however works for whatever, it just spits out its output images
into whatever is set as the current directory for the program, and you can feed an absolute path
into the function as input. 
"""

inputDirectory = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted" #The absolute directory where the input video dataset is stored.
outputDirectory = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted Images" #The absolute directory where you want it to spit out the images.

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
        try:    
            frames_directory = (action_output_directory + "/" + video)
            os.makedirs(frames_directory) #make directory for frames
            os.chdir(frames_directory) #go to directory
    
            video_directory = (inputDirectory + "/" + action + "/" + video)
            FrameCapture(video_directory,video) #Extract frames from video into present directory
        except:
            print("An error occured while splitting " + video)    
        
    print(action + " done...")

print("image extraction complete!")