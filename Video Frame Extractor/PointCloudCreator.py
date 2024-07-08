from os import replace
import cv2
import numpy as np
import math

depth_multiplier = 2 #Can be any number, set it to two as I think it makes pointclouds look better/more accurate
FOV = 26.5 * math.pi/180

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
    POVDepth = math.tan(angle_horizontal)*width_middle
    print(POVDepth)
    ratio1 = 0
    ratio2 = 0
    for point in array:
        magnitude = np.sqrt(np.sum(np.square(np.array([(point[0]-height_middle), (point[1]-width_middle), point[2]]))))
        direction_vector = np.array([point[0],point[1],0])-np.array([height_middle,width_middle,POVDepth])
        direction_vector = direction_vector/np.sqrt(np.sum(np.square(direction_vector)))
        #print(point[0])
        if((point[1]-width_middle)>0):
            ratio1 = ratio1+1
        else:
            ratio2 = ratio2+1

        direction_vector = direction_vector*magnitude#(point[2]+POVDepth)#magnitude #Now this should be the new point

        point[0] = direction_vector[0]
        point[1] = direction_vector[1]
        point[2] = direction_vector[2]

    return array

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

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
#test = cv2.imread("/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted Images/run/50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi/50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi_frame0.jpg")
#print(test.shape)

#test2 = np.load("/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted Depths/run/50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi/50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi_frame0.npy")
#print(test2.shape)
#test2 = np.transpose(test2, (1, 2, 0))
#print(test2.shape)
#test2 = np.resize(test2, (240, 320, 1))
#print(test2.shape)

pointcloud = convert_to_pointcloud("/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted Images/run/50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi/50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi_frame0.jpg",
                            "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted Depths/run/50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi/50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi_frame0.npy")
#print(numpy_to_text(pointcloud))
with open("test.txt", 'w') as output_file:
    output_file.write(numpy_to_text(pointcloud))
    
#print(numpy_to_text(np.zeros((20, 3))))


