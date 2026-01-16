# Background
  I created this project in the summer after my first year of university as a experimental proof-of-concept, but it was originally only for myself so I did not think to make it public on github. 
I have since cleaned it up a bit for public posting (removing hard-coded file directories and replacing them with command line arguements, making this readme and dependencies list, removing some
of the commented out code, and making the pipeline runner that runs all of the programs), but otherwise the code remains largely the same as after my first year. Because of this, the code may not
be as functional or clean as code I would make in the present.

# Longer Explanation of the Project
  The initial goal was to create a program that takes an initial 3D point cloud and the position of an individuals limbs, and then use a form of video generation AI to extrapolate the movements needed to 
perform a given action. This could then theoretically be used in applications such as creating animations or acting as a guide for an artificial intelligence controlling a humanoid robot. This would act as a form
of world model, though I was not aware of the term at the time of making this program. However in order to train such a generative artificial intelligence you would need lots of "3D videos" where poses of individuals
act in a 3D point cloud to use as training data, or you would need a way to generate such training data. That is where this project came in.

  The goal of this project was to make a proof-of-concept program that takes any monocular video of an individual acting, and convert it into one of these 3D point cloud videos. The project then had a secondary goal of filtering out any videos
which do not work for this purpose, such as videos with no people in them, videos that were too blurry, or videos that were to short to see any meaningful action. To achieve this, the project used two computer vision programs:
* [Depth Anything](https://github.com/LiheYoung/Depth-Anything) was used to get the distance of each pixel from the camera.
* [MMPose](https://github.com/open-mmlab/mmpose) was used to get which pixels each limb was located at (using 2D pose estimation), and was used to get the 3D orientation of limbs to eachother (using 3D pose estimation). For both use cases, the demo coco model was used.

The distance of each pixel from the camera was used to make a point cloud from the pixels. The 2D positions of the individuals's head (from the 2D pose estiamtion) was used to determine the individuals position in the point cloud.
The distance of the individuals pose points were used to determine the scale of the individual in the cloud. The individual's scale and head position were then used to place the 3D pose of the individual at a roughly correct scale and position
in the point cloud. This is then done for each frame to create a 3D video.

Given the many steps of this process and the estimations made, there is quite a bit of room for failure to occur. To combat this, several heuristics were employed for detecting faulty frames and point of view changes or prevnting them entirely,
which included but was not limited to:
* Detecting sudden drastic changes in head position (often means either individual was placed in the background for that frame, or a point of view change occured).
* Sudden drastic changes in the orientation of limbs (limb orientation was determined by getting the angles between vectors from one pose point to another, using cross products of these vectors to determine forward from backward)
* Impossibly oriented limbs (using the same orientation of limbs as before, for example the knee going in the opposite direction usually means there is an issue).
* The head would be moved up to 10 pixels in 2D space to be at the closest pixel to the camera (head is often in the foreground relative to surroundings, reduces rate head is incorrectly placed in the background)
* Checking if any part of the pose was further than the mean distance for the pixels from the camera (if any is, usually means incorrectly placed in the background).


First a pass-over would be done removing frames that are determined to definetly faulty rather than point of view changes (ex. impossible joint orientations). Depending on the size of the gap the video would be split into two separate videos
or if the gap was short enough it would be "filled in" by an estimation of what occurs. This filling in would occur if only a few frames were missing, and it would be done by taking the initial vectors and final vectors of the limbs, then using slerp
and linear interpolation to estimate the intermediate limb positions and scales respectivly (i.e. vector direction and magnitude). After this passover, any clips that are too short to bother saving would be discarded before a second pass over is then
performed. This second pass over detects sudden changes that could indicate either a camera angle change or a faulty frame. To distinguish between faulty frames and camera angle changes, the program would determine if it briefly enters this position
before snapping back or if it stays in the new position for an extended period of time. Frames that were detected to be faulty would then again either be used as a splitting point or have the gap filled by an estimation depending on the size of the gap. Frames
that were detected to be rapid changes due to camera angle changes were instead always used as a splitting point. After this second pass over, and clips that were too short to bother saving were thrown away, and any that were left were kept.

  This program was made to complete this process on the HMDB51 data set, though I only ran it on the run and walk tasks to test the concept. On this section of the data set, the filters result in a very high-drop out rate with only a small percentage of the videos
producing any final saved clips. The program was designed to run where the fully body is in view, and runs into issues if only part of their body is visible. It also does not work if two individuals are present in the frame, since it has no means of tracking which
individual is which and only stores one set of pose data any frames with two people in them are marked as faulty. Theoretically these issues are solvable, but I did not do so for this proof of concept. 

# Installation and Dependencies
## Libraries to install
These are the versions of the libraries that worked for me, if you attempt to install them with other versions the program might work but your milage will vary. I advise installing these in the listed order as follows, since that is what worked for me:
* cuda-version==11.8 (may need to install this cuda version on your computer as well not just this library, not completely sure, think its on my computer).
* torch==2.0.0 (from https://download.pytorch.org/whl/cu118)
* torchvision==0.15.1 (from https://download.pytorch.org/whl/cu118)
* openmim
* mmengine==0.10.4 (use mim install)
* mmcv<2.1.0 (use mim install)
* mmdet==3.1.0 (use mim install)
* https://github.com/mattloper/chumpy.git (chumpy should be automatically installed with mmpose but it does not, [due to this issue here](https://github.com/mattloper/chumpy/issues/56), so need to install via git link)
* mmpose>=1.3.1
* numpy==1.24.3 (Install after the above libraries, or they will attempt to overwrite it with a later version which then causes mmpose to not work.)
* huggingface_hub

## Models to download
Run this command in whatever folder you want to store the mmpose coco model in, once the above libraries are downloaded:
mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest .
It will download two files. Do not rename these files, and remember where you stored them. You can move them to a new location as long as you remember where you stored them, keep them together, and dont rename them.
You will need their file location to run the program.

Then download the Depth-Anything model (just download [this repo](https://github.com/LiheYoung/Depth-Anything)) and save the location where you downloaded it (including the Depth-Anything folder itself). You will
need this file location to run the program.

# How to use
IN PROGRESS
