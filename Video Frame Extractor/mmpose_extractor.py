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


#Old args
mmpose_path = "/mnt/c/AI_model/3DMovementModel/Video Pose Estimator/mmpose/"

config = mmpose_path+"td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
checkpoint = mmpose_path+"td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"
device = "cuda:0"

img = mmpose_path+"tests/data/coco/000000000785.jpg"



def main():
    # build the model from a config file and a checkpoint file
    cfg_options = None

    model = init_model(
        config,
        checkpoint,
        device=device,
        cfg_options=cfg_options)

    #WE NEED TO EDIT THIS SO THAT IT LOOPS
    #ALSO WE SHOULD MOVE THIS TO VISUAL STUDIO ENVIRONMENT

    # inference a single image
    batch_results = inference_topdown(model, img)
    results = merge_data_samples(batch_results)
    
    print(results.pred_instances.keypoints) #THIS LINE HERE WAS ADDED BY ME, it prints the keypoints, though for some reason there is an empty dimensions beforehand in the array.
    #Each keypoint is stored as a pair, the pair is a 2 member list
    print(results.pred_instances.keypoints[0]) #This compensates for the extra dimension
    #I suspect the extra dimension implies it is possible to do batches of images, where each first dimension layer is one of the batch members.
    

if __name__ == '__main__':
    main()

