#Used this: https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/

import cv2 
import sys
import os
import shutil
  
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
        except KeyboardInterrupt:
            exit()
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
args = sys.argv[1:]
if(len(args)==0):
    print("Frame extractor needs absolute directory of extracted HMDB51 files as arguement.")
    sys.exit(-1)
elif (len(args)>1):
    print("Frame extractor Ignoring additional/extra arguements.")

inputDirectory = args[0] #The absolute directory where the input video dataset is stored.
homeDirectory = os.getcwd()
outputDirectory = os.path.join(homeDirectory, "extracted_images") #Spit out the images in a folder in project.

os.chdir(inputDirectory) #First, go to the input directory and get its members
actions = os.listdir() #got its members (and in HMDB51 also corresponds to main action)

if(not os.path.exists(outputDirectory)):
        os.makedirs(outputDirectory) #Create output directory if it does not exist
os.chdir(outputDirectory) #now, create corresponding output directories for each action

for action in actions: 
    action_path = os.path.join(outputDirectory, action)
    if(not os.path.exists(action_path)):
        os.makedirs(action_path) #Create action output directory if it does not exist


for action in actions: #now that we have created the nessecary directories, we can extract the images into them
    print("processing " + action + "...")

    os.chdir(os.path.join(inputDirectory, action))
    videos = os.listdir() #get a list of the input videos.
    
    
    action_output_directory = os.path.join(outputDirectory, action)
    if(not os.path.exists(action_output_directory)):
        os.makedirs(action_output_directory) #Create action output directory if it does not exist
    os.chdir(action_output_directory) #return to output directory for action
    
    for video in videos:
        frames_directory = os.path.join(action_output_directory, video)
        if os.path.isdir(frames_directory): #Wipe old frames directory  contents.
            shutil.rmtree(frames_directory)

        os.makedirs(frames_directory) #make directory for frames
        os.chdir(frames_directory) #go to directory
    
        video_directory = os.path.join(inputDirectory, action, video)
        FrameCapture(video_directory,video) #Extract frames from video into present directory
        
    print(action + " done...")

print("image extraction complete!")