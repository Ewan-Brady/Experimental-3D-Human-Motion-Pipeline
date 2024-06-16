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
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
        # Saves the frames with frame-count 
        cv2.imwrite(video_name+("_frame%d.jpg" % count), image) 
  
        count += 1
        
#outputDirectory = ""
#os.chdir(outputDirectory) #Change directory to output directory, use absolute path for input.

FrameCapture("test.mp4", "test")