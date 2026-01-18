#Environment is TestEnv.
import os
import shutil
import sys

args = sys.argv[1:]
if(len(args) < 3):
    print("Depth frame extractor needs absolute directory of extracted database files, absolute directory of Depth-Anything, and an absolute directory to place the output folder in as arguements 1, 2, and 3 respectivly.")
    sys.exit(-1)
elif (len(args)> 3):
    print("Depth frame extractor ignoring additional/extra arguements.")

inputDirectory = args[0] #The absolute directory where the input video dataset is stored.
depth_anything_directory = args[1] #The absolute directory of Depth-Anything.
homeDirectory = args[2]
outputDirectory = os.path.join(homeDirectory, "estimated_depths") #Spit out the depths in a folder in project.

sys.path.append(depth_anything_directory)
os.chdir(depth_anything_directory)

#Do imports in the correct directory
import numpy as np
import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


#encoders = ['vits', 'vitb', 'vitl'] vits is fast enough for real time, vitl is quite slow. vitb is a decently fast middle ground
#We will try making a vitl version for now, but if it ends up too painfully slow we can go with a vitb
encoder = 'vitl'
video_path = 'test.avi'


margin_width = 50
caption_height = 60

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

depth_anything = DepthAnything.from_pretrained(os.path.join("LiheYoung", "depth_anything_{}14".format(encoder))).to(DEVICE)

total_params = sum(param.numel() for param in depth_anything.parameters())
print('Total parameters: {:.2f}M'.format(total_params / 1e6))

depth_anything.eval()

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])


# Define the codec and create VideoWriter object

def save_video_depth_frames(directory, video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []

    i = 0

    if(os.path.exists(directory + ".npy")): #Skips finished files to resume.
        #os.system('cls' if os.name == 'nt' else 'clear')
        print(("Skipping:   " + "{:05d}".format(i)), end = '\r')
        i=i+1
        return #Directory exists, skip


    while cap.isOpened():
        target_location = directory

        ret, raw_image = cap.read()
        if not ret:
            break

        print(("Processing: " + "{:05d}".format(i)), end = '\r')

        raw_image = cv2.resize(raw_image, (640, 480))

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    
        h, w = image.shape[:2]
    
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    
        with torch.no_grad():
            depth = depth_anything(image)

        frames.append(depth.cpu().numpy())
        i=i+1
    np.save(target_location, np.stack(frames))
    cap.release()
#COPYPASTED MOSTLY FROM FRAME EXTRACTOR
"""
The below iterator is made to extract from the HMDB51 dataset's directory structre. 
The frame extraction function however works for whatever, it just spits out its output images
into whatever is set as the current directory for the program, and you can feed an absolute path
into the function as input. 
"""


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
        video_directory = os.path.join(inputDirectory, action, video)
        target_directory = os.path.join(action_output_directory,video)
        save_video_depth_frames(target_directory, video_directory) #Extract frames from video into present directory
        
    print(action + " done...")

print("depth extraction complete!")