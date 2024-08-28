#Conda environment VideoFrameExtractor
from os import replace
from re import A
from turtle import forward
import cv2
import numpy as np
import math
import os
import sys
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

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
    image = cv2.resize(image, (320, 240))
    image = image * 255
    #print(image.shape)


    depth = np.load(depth)
    depth = np.transpose(depth, (1, 2, 0))
    depth = cv2.resize(depth, (320, 240))
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
    #return array
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
                
        #point[2] = direction_vector[2] #THIS WAS COMMENTED OUT BEFORE

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
Given the 2D pixelspace skeleton file, the 3D nonpixelspace skeleton file, the pointcloud data,
and the directory where the depth data is stored, split the data into usable training segments and process it
for training (the processing is done by pose_extract_3d, but you need to split the data first.)
"""
minimum_video_frames = 20 #discard all data clips that are below this frame length. 
fill_in_cutoff = 5 #Attempt to Fill in all gaps of this length or lower.
def process_data(skeleton_file_2d, skeleton_file_3d, depth_directory, point_clouds, video_name):
    """
    load skseleton data from file
    """
    skeleton_frames_2d = np.load(skeleton_file_2d)
    skeleton_frames_3d = np.load(skeleton_file_3d)

    """
    mismatch check
    """
    if not (len(skeleton_frames_2d) == len(skeleton_frames_3d) == len(point_clouds)):
        print((len(skeleton_frames_2d) == len(skeleton_frames_3d) == len(point_clouds)))
        raise Exception("Mismatch between total frame amounts: " + str(len(skeleton_frames_2d)) + 
                        ", " + str(len(skeleton_frames_3d)) + 
                        ", " + str(len(point_clouds)))


    num_frames = len(skeleton_frames_2d)
    """
    Iterate over the frames to check which are faulty, and get the depths for each frame
    """
    depth_frames = []
    faulty_frames = []
    for i in range(num_frames):
        #Check for all negative one to indicate flawed frame.
        if(np.array_equal(skeleton_frames_2d[i],np.full(skeleton_frames_2d[i].shape,-1)) or 
           np.array_equal(skeleton_frames_3d[i],np.full(skeleton_frames_3d[i].shape,-1))):
            faulty_frames.append(i)
        
        #Copypasted from pointcloud functions, use this to get depth for frame
        depth_frame = (depth_directory+ "/" + video_name + "_frame" + str(i) + ".npy")

        depth_frame = np.load(depth_frame)
        depth_frame = np.transpose(depth_frame, (1, 2, 0))
        depth_frame = cv2.resize(depth_frame, (320, 240))
        depth_frame = np.expand_dims(depth_frame, axis = 2)
        depth_frame = np.max(depth_frame)-depth_frame
        depth_frame = depth_frame * depth_multiplier

        depth_frame = np.transpose(depth_frame, (1,0,2))
        depth_frame = np.flip(depth_frame, axis = 1)

        depth_frames.append(depth_frame)
    
    depth_frames = np.stack(depth_frames)
    
    """
    mismatch check
    """
    if(len(depth_frames) != num_frames):
        raise Exception("Mistmatch between depth frames " + str(len(depth_frames)) + 
                        " and total frames " + str(len(num_frames)))
    
    """
    Process data for each frame and concatenate into clips,
    clips are separated by the faulty frames
    """
    previous_frame_faulty = True #Having this as true makes it so that it makes a new first clip.
    num_previous_faulty_frames = fill_in_cutoff+1 #Must always start greater than fill-in cutoff so it makes new first clip
    data = []
    clip_gaps = []
    for i2 in range(num_frames):
        if(i2 in faulty_frames):
            previous_frame_faulty = True #Frame is faulty dont process
            num_previous_faulty_frames += 1
        else:
            pose_points_3d, pose_angles_3d, point_cloud, head_pos = pose_extract_3d(skeleton_frames_2d[i2], skeleton_frames_3d[i2],
                                                                      depth_frames[i2], point_clouds[i2]) #Process frame
            head_angle = pose_angles_3d[0]
            """
            Before adding it to the data, Check by depth, if any pose point is below the mean depth it is probally in the background.
            If it is in the background the frame is faulty and is discarded.
            """
            mean_depth = np.sum(point_cloud,axis=0)[2]/len(point_cloud)
            current_frame_faulty = False
            for i in pose_points_3d:
                if i[2] > mean_depth:
                    previous_frame_faulty = True #There is a point in the background, this frame is faulty.
                    current_frame_faulty = True
                    num_previous_faulty_frames += 1
                    break
            if current_frame_faulty: #Frame was marked faulty in the loop, so move on to the next frame 
                continue
            
            """
            This line rotates all points rotate relative to the head, other than on the up axis (y)
            If you dont want it to do that comment it out.
            It also includes updating the angles to reflect the new points.
            """
            pose_points_3d, point_cloud, pose_angles_3d[:2] = rotate_about_head(pose_points_3d,point_cloud,pose_angles_3d[:2])            

            """
            In this area, we check if the frame is faulty based on odd joints such as backwards knees
            """
            current_frame_faulty = odd_joints_check(pose_angles_3d)
            if current_frame_faulty:
                previous_frame_faulty = True
                num_previous_faulty_frames += 1
                continue

            if(previous_frame_faulty): #Make a new clip with gathered frame data, previous frames were marked faulty.
                data.append([np.array([pose_points_3d]),
                             np.array([pose_angles_3d]),
                             np.array([point_cloud]),
                             np.array([head_pos]),
                             np.array([head_angle])])
                clip_gaps.append(num_previous_faulty_frames)
                num_previous_faulty_frames = 0
                previous_frame_faulty=False
                """
            
            elif(previous_frame_faulty): #A small enough amount of previous frames were marked faulty to do a fill-in
                
                This segment here allows for filling in of short segments of faulty frames.
                
                One possible concern is this and later fill-in steps could all occur around the same area, resulting in
                too much data being fill in. We will see if this ends up being a large issue.
                
                index = (len(data)-1) #Append gathered frame data to pre-existing clip.
                for i in range(num_previous_faulty_frames+1): #Concatenate the number of faulty frames +1, all but last will be replaced with fill-in
                    data[index][0] = np.concatenate([data[index][0], np.array([pose_points_3d])], axis=0)
                    data[index][1] = np.concatenate([data[index][1], np.array([pose_angles_3d])], axis=0)
                    data[index][2] = np.concatenate([data[index][2], np.array([point_cloud])], axis=0)
                    data[index][3] = np.concatenate([data[index][3], np.array([head_pos])], axis=0)
                    data[index][4] = np.concatenate([data[index][4], np.array([head_angle])], axis=0)
                fill_in_slerp(data[index],(len(data[index][0])-2-num_previous_faulty_frames),(len(data[index][0])-1)) #Fill in faulty frames
                
                num_previous_faulty_frames = 0
                previous_frame_faulty=False"""
            else:
                index = (len(data)-1) #Append gathered frame data to pre-existing clip.
                data[index][0] = np.concatenate([data[index][0], np.array([pose_points_3d])], axis=0)
                data[index][1] = np.concatenate([data[index][1], np.array([pose_angles_3d])], axis=0)
                data[index][2] = np.concatenate([data[index][2], np.array([point_cloud])], axis=0)
                data[index][3] = np.concatenate([data[index][3], np.array([head_pos])], axis=0)
                data[index][4] = np.concatenate([data[index][4], np.array([head_angle])], axis=0)
    clip_gaps.append(fill_in_cutoff+1) #Indicates cant fill in gap between final element and the nonexistant element after that.

    """
    This function attempts to connect frames that are only separated by a small amount of faulty frames, 
    ensuring that it is not a camera angle cut before doing so.
    """
    if(len(data) > 0):
        data, head_data, pose_data, angle_data, head_angle_data = seal_nicks(data, clip_gaps, fill_in_cutoff)
    else:
        return []
    
    """
    Remove too-short clips
    """
    data = remove_short_clips(data, minimum_video_frames)
    
    """
    process clips for errors.
    
    This process does two things: eliminates/masks small glitches in clips, and splits clips where it thinks
    a camera angle jump occured. 
    """
    processed_data = []
    for clip in data:
        clips = process_clip(clip, fill_in_cutoff, head_data, pose_data, angle_data, head_angle_data)
        processed_data = processed_data + clips
    data = processed_data
    
    """
    Remove too-short clips again.
    """
    data = remove_short_clips(data, minimum_video_frames)
    
    """
    Return the data

    data is in a nx3 matrix. The first dimension is the various clips, and the second dimension contains
    the 3d point frames, 3d angle frames, and point cloud frames for each clip.
    """
    return data

"""
Given the pose angle quaternions, returns False if no odd joints (such as backwards knees) are
detected and returns True if an odd joint is detected.

Currently just does knees, may implement more in the future.
"""
def odd_joints_check(pose_quaternions):
    knee_cutoff_threshhold = -0.01 #If the knee angle x value (the value other than w that varies) is anywhere below this, is a faulty angle
    if(pose_quaternions[10][0] < knee_cutoff_threshhold or pose_quaternions[11][0] < knee_cutoff_threshhold):
        return True #Odd knee joint detected
    return False #Made it to the end with nothing odd detected.

"""
rotates processed pose_points_3d, pointcloud, and head angles of pose_angles_3d about the head orientation, while
still keeping absolute up vector.
"""
absolute_upward_vector = np.array([0,1,0]) #used in slerp function as well, Just points straight up relative to coordanate system.
def rotate_about_head(pose_points_3d, pointcloud, head_angles):
    #calculate forward vector from averaging out vector to eyes and vector to mouth 
    forward_vector_1 = pose_points_3d[9]-pose_points_3d[17] #Tip-tail, mouth-head
    forward_vector_2 = pose_points_3d[10]-pose_points_3d[17] #Tip-tail, eyes-head
    forward_vector = forward_vector_1+forward_vector_2
    forward_vector[1] = 0 #remove vertical component
    forward_vector = forward_vector/np.sqrt(np.dot(forward_vector,forward_vector))

    #Upward vector is made perpendicular in the function, causing it to go straight up
    head_rotation = (R.from_quat(orientation_quaternion(forward_vector,absolute_upward_vector))).inv()
    head_rotation_matrix = head_rotation.as_matrix()

    for i in range(len(head_angles)):
        old_orientation = (R.from_quat(head_angles[i])).as_matrix()
        head_angles[i] = (R.from_matrix(np.matmul(head_rotation_matrix,old_orientation))).as_quat()

    pose_points_3d = head_rotation.apply(pose_points_3d)
    pointcloud[:,:3] = head_rotation.apply(pointcloud[:,:3])

    return pose_points_3d, pointcloud, head_angles

"""
Attempts to fill in for initial faulty frames in primary data processing I.E. before process_clip is called,
faulty frames caused by the frame itself rather than its relation to other frames.

The process goes about as follows:
    -First, feed all data clips into the function, along with the tracked gap lengths 
    -Then, calculate total means and standard deviations based on the collective gaps of all of the clips 
    -Then, for each gap length that is small enough that it could be filled in, check if the head_gap and size_gap are 
    below the z_score threshhold. If they are, mark the two clips to be merged. 
    -Finally, merge all clips marked to be merged, and return the clips. 
"""
primary_z_score_cutoff_head = 3 #The z-score cutoff to determine which head position changes are large enough jumps to be anomalies
primary_z_score_cutoff_size = 3 #The z-score cutoff to determine which body position changes are large enough jumps to be anomalies
primary_z_score_cutoff_angle = 3 #The z-score cutoff to determine which angle changes are large enough jumps to be anomalies
primary_z_score_cutoff_absolute_angle = 3 #The z-score cutoff to determine which absolute head orientation changes are large enough to be anomalies
def seal_nicks(data, data_gaps, fill_in_cutoff):
    head_gaps = []
    size_gaps = []
    pose_sizes = []
    angle_gaps = []
    head_angle_gaps = []
    for clip in data:
        head_gaps_new = calculate_distance_gaps(clip[3])
        head_gaps = head_gaps + head_gaps_new #concatenate new head distance gaps
        
        pose_sizes_new, size_gaps_new = calculate_size_gaps(clip[0])
        size_gaps = size_gaps + size_gaps_new #concatenate new size gaps
        
        pose_sizes.append(pose_sizes_new) #Append pose_sizes
        

        angle_gaps_new = calculate_angle_gaps(clip[1]) #include absolute head orientation
        if(len(angle_gaps) == 0): #For the first time
            angle_gaps = angle_gaps_new
        else: #For following times
            if(len(angle_gaps_new) != 0):
                angle_gaps = np.concatenate([angle_gaps, angle_gaps_new], axis=1)
                
        head_angle_gaps_new = list(calculate_individual_angle_gaps(clip[4]))
        head_angle_gaps = head_angle_gaps+head_angle_gaps_new

    if(len(head_gaps) != len(size_gaps)):
        raise Exception("Mistmatch between size_gaps and head_gaps length in nick-sealing stage.")
    if((len(head_gaps) == 0) or (len(size_gaps) == 0)):
        """
        In this case there isnt a single, usable frame that isnt isolated. While in theory this data could be used,
        we cannot verify camera cuts and so it is to be discarded
        """
        return [], (0,0), (0,0), (0,0), (0,0)
    
    mean_head, standard_deviation_head = deviations_and_mean(head_gaps)
    mean_sizes, standard_deviation_sizes = deviations_and_mean(size_gaps)
    mean_head_angle, standard_deviation_head_angle = deviations_and_mean(head_angle_gaps)

    #Get a list of the mean and deviations for the gaps in each angle
    angle_gap_means = []
    angle_gap_deviations = []
    for i in angle_gaps:
        angle_gap_mean, angle_gap_deviation = deviations_and_mean(i)
        
        angle_gap_means.append(angle_gap_mean)
        angle_gap_deviations.append(angle_gap_deviation)

    gaps_to_merge = []
    #data_gaps[0] is from -1 to 0, data_gaps[1] is from 0 to 1, I.E. data gaps index represents leading data piece. 
    for i in range(len(data_gaps)):
        if data_gaps[i] <= fill_in_cutoff:
            initial = data[i-1]
            final = data[i]
            initial_index = len(initial[3])-1 #The final index of the initial clip
            if fill_in_zscore_check(initial,final,initial_index,0,
                                     (mean_head, standard_deviation_head, primary_z_score_cutoff_head),
                                     (mean_sizes, standard_deviation_sizes, primary_z_score_cutoff_size), 
                                     (angle_gap_means, angle_gap_deviations, primary_z_score_cutoff_angle), 
                                     (mean_head_angle, standard_deviation_head_angle, primary_z_score_cutoff_absolute_angle)):
                gaps_to_merge.append(i) #Passed z score tests, likely not a camera angle jump, order a fill in.  
                
    new_data = []
    for i in range((len(data_gaps)-1)):
        if i in gaps_to_merge: #Merge ordered with this piece and previous.
            final_index = len(new_data)-1
            
            start_value = len(new_data[final_index][0])-1 #the final frame of the initial clip
            end_value = data_gaps[i] + start_value + 1
            
            for i2 in range(len(new_data[final_index])): #First concatenate each data piece with a set of 0s in-between of specified size.
                shape = list(data[i][i2][0].shape) #Get shape per frame
                shape = [data_gaps[i]] + shape

                filler_array = np.zeros(shape)

                new_data[final_index][i2] = np.concatenate([new_data[final_index][i2], filler_array, data[i][i2]])
            new_data[final_index] = fill_in_slerp(new_data[final_index],start_value,end_value) #Fill in the numpy zero array fillers via slerp fill-in method.
            
        else:
            new_data.append(data[i]) #No merge ordered for this leading piece with previous, so append it.

    return (new_data, (mean_head, standard_deviation_head), (mean_sizes, standard_deviation_sizes), 
            (angle_gap_means, angle_gap_deviations), (mean_head_angle, standard_deviation_head_angle))

"""
Calculates a list of distance values for adjascent points given an array of points
"""
def calculate_distance_gaps(positions_list):
    head_gaps = [] #This is a list of the distances between frames
    for i in range((len(positions_list)-1)):
        distance_vector = positions_list[i]-positions_list[(i+1)]
        distance = np.sqrt(np.sum(np.square(distance_vector)))
        head_gaps.append(distance)
        
    return head_gaps

"""
#For an array of pose data 3d frames, calculates the pose size gap values. Also returns pose sizes.
"""
def calculate_size_gaps(pose_data_3d):
    pose_sizes = calculate_pose_sizes(pose_data_3d)
                 
    #Now find the gaps between sizes for each frame
    size_gaps = [] #This is a list of the distances between frames
    for i in range((len(pose_sizes)-1)):
        gap = abs(pose_sizes[i]-pose_sizes[(i+1)])
        size_gaps.append(gap)
        
    return pose_sizes, size_gaps

"""
Calculates values representing pose size for given pose data, used in multiple functions
"""
def calculate_pose_sizes(pose_data_3d):
    pose_sizes = []
    for i in pose_data_3d:
        point_distances = []
        covered_points = []
        for i2 in range(len(i)):
            covered_points.append(i2)
            for i3 in range(len(i)):
                if not i3 in covered_points:
                    distance_vector = i[i2]-i[i3]
                    distance = np.sqrt(np.sum(np.square(distance_vector)))
                    point_distances.append(distance)
        mean_distance = sum(point_distances)/len(point_distances)
        pose_sizes.append(mean_distance)
        
    return pose_sizes

"""
Given 3d pose angle data, get values that represent the changes in angles
Gives a (number of angles)x(frame_gaps) matrix of scalar values that represent the opposite of the W value for the quaternion
that causes a change from 1 frame to another (I.E. if the w value of the quaternion is 0 then it will be 1, ect)
"""
def calculate_angle_gaps(pose_angles_3d):
    all_angle_gaps = []
    for i in range(len(pose_angles_3d[0])): #Iterate over angles
        all_angle_gaps.append(calculate_individual_angle_gaps(pose_angles_3d[:,i])) #Append angle gaps fot the desired angle
        """
        #Moved this to external function:
        angle_gaps = [] #Represents the angle gaps for the 

        for initial_frame_index in range((len(pose_angles_3d)-1)): #Iterate over each frame except the last
            initial_frame = pose_angles_3d[initial_frame_index][i] #Get initial angle
            final_frame = pose_angles_3d[initial_frame_index+1][i] #Get final angle
            
            intermediate_quaternion = quaternion_between_quaternions(initial_frame,final_frame)
            
            change_value = abs(intermediate_quaternion[3]-1) #Subtract 1 from W and get absolute value to get value that represents change in angle.
            angle_gaps.append(change_value) #Append to gaps between each frame
            
        if(len(angle_gaps) > 1):
            angle_gaps = np.array(angle_gaps) #Concatenate
            all_angle_gaps.append(angle_gaps) #Append to gaps for each angle
        else:
            all_angle_gaps.append(angle_gaps) #Handles case of only one gap.
        """    
    
    all_angle_gaps = np.stack(all_angle_gaps)
    return all_angle_gaps
 
"""
Calculate the gaps for an individual angle over a number of frames
"""
def calculate_individual_angle_gaps(angle_list):
    angle_gaps = [] #Represents the angle gaps for the 

    for initial_frame_index in range((len(angle_list)-1)): #Iterate over each frame except the last
        initial_frame = angle_list[initial_frame_index] #Get initial angle
        final_frame = angle_list[initial_frame_index+1] #Get final angle
        angle_gaps.append(scalar_quaternion_gap(initial_frame,final_frame)) #Append to gaps between each frame
            
    if(len(angle_gaps) > 1):
        angle_gaps = np.array(angle_gaps) #Concatenate
        return angle_gaps #Return gaps
    else:
        return angle_gaps #Handles case of only one gap.
"""
Returns a scalar value representing the "magnitude" of a difference between two quaternion rotations, used in other functions
"""
def scalar_quaternion_gap(quaternion1, quaternion2):
    intermediate_quaternion = quaternion_between_quaternions(quaternion1,quaternion2)

    change_value = abs(intermediate_quaternion[3]-1) #Subtract 1 from W and get absolute value to get value that represents change in angle.
    
    return change_value

"""
#Calculates the standard deviation and mean for a list of numbers
"""
def deviations_and_mean(numbers_list):
    #Calculate the mean
    mean = sum(numbers_list)/len(numbers_list)
    
    #Calculate standard deviation
    sum_for_deviation = 0
    for i3 in numbers_list:
        sum_for_deviation = sum_for_deviation+((i3-mean)**2)
    
    deviation = (sum_for_deviation/(len(numbers_list)-1))**0.5
    
    return mean, deviation #Return

"""
Given the z score parameters we check, does the gap meet z score requirements to be flagged as a significant gap? 

Feed in initial data, final data, initial frame in initial data, final frame in final data, and tuples containing:
Mean
Standard Deviation
Z score cutoff

for each z score parameter.
"""
def fill_in_zscore_check(initial_data, final_data, initial_index, final_index, head_info, size_info, relative_angle_info, absolute_angle_info):
    #Unpack head_info, size_info, relative_angle_info, and absolute_angle_info
    mean_head, standard_deviation_head, head_zscore_cutoff = head_info
    mean_sizes, standard_deviation_sizes, size_zscore_cutoff = size_info
    angle_gap_means, angle_gap_deviations, relative_angle_zscore_cutoff = relative_angle_info
    mean_head_angle, standard_deviation_head_angle, absolute_angle_zscore_cutoff = absolute_angle_info

    #Head gap and size gap zscore
    head_gap_vector = final_data[3][final_index]-initial_data[3][initial_index] #Calculate head gap
    head_gap = np.sqrt(np.dot(head_gap_vector,head_gap_vector)) 
            
    pose_sizes_initial = calculate_pose_sizes(initial_data[0])
    pose_sizes_final = calculate_pose_sizes(final_data[0])

    size_gap = abs(pose_sizes_final[final_index]-pose_sizes_initial[initial_index]) #Gap in size of first frame final clip and final frame initial clip
  
    z_score_head = (head_gap-mean_head)/standard_deviation_head
    z_score_size = (size_gap-mean_sizes)/standard_deviation_sizes
    #Relative angle zscores
    angle_z_score_pass = True
    for angle_index in range(len(angle_gap_means)): #Iterate through each angle and get z score for its gap.
        #Get scalar value representing the gap
        angle_gap = scalar_quaternion_gap(initial_data[1][initial_index][angle_index],final_data[1][final_index][angle_index])
        #Calculate z score
        z_score_angle = (angle_gap-angle_gap_means[angle_index])/angle_gap_deviations[angle_index]
                
        if(z_score_angle >= relative_angle_zscore_cutoff):
            angle_z_score_pass = False #z score exceeds cutoff, do not merge.
            break

    #Absolute head angle zscore.
    #Get scalar value representing the gap
    angle_gap = scalar_quaternion_gap(initial_data[4][initial_index],final_data[4][final_index]) 
    #Calculate z score
    z_score_absolute_angle = (angle_gap-mean_head_angle)/standard_deviation_head_angle

    return (z_score_head < head_zscore_cutoff and z_score_size < size_zscore_cutoff 
            and angle_z_score_pass and z_score_absolute_angle < absolute_angle_zscore_cutoff)
    
"""
This function identifies issues based on changes in head position.
The main issues it identifies are:
     -Brief "islands" where the head is in a drastically different position, likely caused by being placed in background
     instead of foreground. In this case it tries to mask the issue.
     -Jumps from the head hanging out in one position to hanging out in another, likely caused by a camera angle change.
     In this case in splits the clip.
"""
secondary_z_score_cutoff_head = 3 #The z-score cutoff to determine which head position changes are large enough jumps to be anomalies
secondary_z_score_cutoff_size = 3 #The z score cutoff to determine which body size changes are large enough jumps to be anomalies
secondary_z_score_cutoff_angle = 3 #The z-score cutoff to determine which angle changes are large enough jumps to be anomalies
secondary_z_score_cutoff_absolute_angle = 3 #The z-score cutoff to determine which head angle changes are large enough jumps to be anomalies
def process_clip(data, fill_in_cutoff, head_info, pose_info, angle_info, head_angle_info):
    """
    First find gap indicators from the head suddenly changing position.
    """    
    head_gaps = calculate_distance_gaps(data[3]) #This is a list of the distances between frames
        
    #mean_head, standard_deviation_head = deviations_and_mean(head_gaps)
    mean_head, standard_deviation_head = head_info

    """
    Now detect gaps based on sudden increases in the average distances between points, I.E. changes in the size of the individual.
    """
    pose_sizes, size_gaps = calculate_size_gaps(data[0])
    #mean_sizes, standard_deviation_sizes = deviations_and_mean(size_gaps)
    mean_sizes, standard_deviation_sizes = pose_info

    
    #If size_gaps and head_gaps are different lengths something went wrong.
    if len(size_gaps) != len(head_gaps):
        raise Exception("Different number of head gaps than pose size gaps.")
    

    """
    Now detect gaps based on sudden changes in angles
    """
    angle_gap_means, angle_gap_deviations = angle_info
    
    mean_head_angle, standard_deviation_head_angle = head_angle_info
    
    """
    Now calculate z scores for both head gaps and size gaps, and angle gaps, and assign gap indicators based on z score.
    """
    gap_indicators = []
    #Calculate z scores and use to find large jumps in data.
    gap_indicators.append(-1) #First and final gaps should also be marked 
    for i4 in range(len(head_gaps)):
        initial_index = i4
        final_index = i4+1

        if not fill_in_zscore_check(data,data,initial_index,final_index,
                                (mean_head, standard_deviation_head, secondary_z_score_cutoff_head),
                                (mean_sizes, standard_deviation_sizes, secondary_z_score_cutoff_size), 
                                (angle_gap_means, angle_gap_deviations, secondary_z_score_cutoff_angle), 
                                (mean_head_angle, standard_deviation_head_angle, secondary_z_score_cutoff_absolute_angle)):
            gap_indicators.append(i4)
                
    gap_indicators.append(len(head_gaps))# Mark final gap
    
    """
    Now that we have identified the frames with gaps we can identify whether it is a brief change in position or
    a permanent one, and implement the corresponding changes.
    """
    #Use the largest continious length as the baseline for which is correct and which is anomaly.
    contious_segment_lengths = []
    #This keeps track of the starting and ending points for each index in continious_segment_lengths for later usage
    segment_starts_ends = {} 
    i = 0
    for initial_index in range((len(gap_indicators)-1)):
        initial = gap_indicators[initial_index]+1
        final = gap_indicators[initial_index+1]
        length = final-initial+1
        contious_segment_lengths.append(length)
        
        segment_starts_ends[i] = [initial,final]
        i+=1
    
    """
    By uncommenting this you can test how the data splits in varying ways if different gap indictators are ordered.
    contious_segment_lengths = [2,13,1,6,2]
    segment_starts_ends = {}
    segment_starts_ends[0] = [0,1]
    segment_starts_ends[1] = [2,14]
    segment_starts_ends[2] = [15,15]
    segment_starts_ends[3] = [16,21]
    segment_starts_ends[4] = [22,23]
    """
    
    #This gets +1 for odd indexes and 0 for even indexes
    #It identifies whether the maximum segments index is even or odd.
    max_index =  contious_segment_lengths.index(max(contious_segment_lengths))
    odd_or_even = max_index%2 
    
    """
    This loop identifies whether the gaps between each segment are caused by a camera-jump or a brief/fill-innable error.
    
    Process:
    Loop through pairs
    The cause of a gap can be one of two things:
    1. It is a camera cut: 
                   -In this case the length of the opposite parity segment will be small
    2. It is a background/brief snap glitch:
                   -In this case the length of the "odd" segment will be small, and the parity of the opposite parity segment should be swapped
    
    Finally, if it is short and the first or last segment then it must always be deleted as there is insufficent information to fill it in.
                   
    two loops, one going forward one going backward. 
    """
    indexes_to_fill_in = [] #Segment indexes in this liast are marked to be filled in by approximation.
    indexes_to_split = [] #Segment indexes in this list are marked to have a split occur between them and the previous segment.
    remove_last_segment = False
    remove_first_segment = False
    #Forward loop
    loop_current_parity = odd_or_even+1 #Gets a number of the opposite parity of the index of the max index.
    for segment in range(max_index,(len(contious_segment_lengths))):
        #If loop_odd_or_even_1 is odd this triggers when segment is an odd number,
        #If loop_odd_or_even_1 is even this triggers when segment is an even number
        if((segment+loop_current_parity)%2==0):
            if(contious_segment_lengths[segment] > fill_in_cutoff):
                indexes_to_split.append(segment)
                loop_current_parity += 1 #Swap the parity being checked.
            elif (segment == (len(contious_segment_lengths)-1)): 
                remove_last_segment = True #Last frame is a short glitch, insufficent information to fill it in
            else:
                start_and_end = segment_starts_ends[segment]
                
                if fill_in_zscore_check(data,data,(start_and_end[0]-1),(start_and_end[1]+1),
                                         (mean_head, standard_deviation_head, secondary_z_score_cutoff_head),
                                         (mean_sizes, standard_deviation_sizes, secondary_z_score_cutoff_size), 
                                         (angle_gap_means, angle_gap_deviations, secondary_z_score_cutoff_angle), 
                                         (mean_head_angle, standard_deviation_head_angle, secondary_z_score_cutoff_absolute_angle)):
                    indexes_to_fill_in.append(segment)
                else:
                    indexes_to_split.append(segment)
                    loop_current_parity += 1 #Swap the parity being checked.
                
    
    #Backward loop
    loop_current_parity = odd_or_even+1 #Reset parity for reverse loop
    for segment in range(max_index,-1,-1):
        #If loop_odd_or_even_1 is odd this triggers when segment is an odd number,
        #If loop_odd_or_even_1 is even this triggers when segment is an even number
        if((segment+loop_current_parity)%2==0):
            if(contious_segment_lengths[segment] > fill_in_cutoff):
                indexes_to_split.append((segment+1))
                loop_current_parity += 1 #Swap the parity being checked.
            elif (segment == 0):
                remove_first_segment = True #First frame is a short glitch, insufficent information to fill it in.
            else:
                start_and_end = segment_starts_ends[segment]
                
                if fill_in_zscore_check(data,data,(start_and_end[0]-1),(start_and_end[1]+1),
                                         (mean_head, standard_deviation_head, secondary_z_score_cutoff_head),
                                         (mean_sizes, standard_deviation_sizes, secondary_z_score_cutoff_size), 
                                         (angle_gap_means, angle_gap_deviations, secondary_z_score_cutoff_angle), 
                                         (mean_head_angle, standard_deviation_head_angle, secondary_z_score_cutoff_absolute_angle)):
                    indexes_to_fill_in.append(segment)
                else:
                    indexes_to_split.append(segment)
                    loop_current_parity += 1 #Swap the parity being checked.

    indexes_to_fill_in.sort()
    indexes_to_split.sort()
    
    """
    Use indexes_to_fill_in, indexes_to_split, and segment_starts_ends to order and execute splits,
    fill-ins, and frame removals as nessecary. 
        -Remember to account for the fact that initial and final frames can not be filled in.
    """
    #First, do fill-ins.
    for index in indexes_to_fill_in:
        if(index != 0 and index != (len(contious_segment_lengths)-1)): #Check that not start or end frame as those cant be filled in.
            start_and_end = segment_starts_ends[index]
            data = fill_in_slerp(data,(start_and_end[0]-1),(start_and_end[1]+1)) 

    #Second, remove front and rear segments if nessecary
    covered_frames = 0 #Number of beginning indexes removed from data. 
    if remove_last_segment:
        final_segment_start = segment_starts_ends[(len(contious_segment_lengths)-1)][0]
        for i in range(len(data)):
            data[i] = data[i][:final_segment_start,...]
    if remove_first_segment:
        second_segment_start = segment_starts_ends[1][0]
        for i in range(len(data)):
            data[i] = data[i][second_segment_start:,...]
        covered_frames = contious_segment_lengths[0] #First segment size is equal to number of frames removed.
    
    #Finally, split into separate lists as ordered
    to_return = []
    for end_segment in indexes_to_split:
        end_segment_start = segment_starts_ends[end_segment][0]-covered_frames
        
        previous_segment = []
        for i in data: #Add cut off segment to info to be returned
            previous_segment.append(i[:end_segment_start,...])
        to_return.append(previous_segment)
        
        covered_frames += contious_segment_lengths[(end_segment-1)]
        
        for i in range(len(data)): #remove cut off segment from data.
            data[i] = data[i][end_segment_start:,...]
    to_return.append(data)
    
    """
    Can uncomment this to see how it split
    print(contious_segment_lengths)
    for i in to_return:
        print(str(i[0].shape) + " " + str(i[1].shape) + " " + str(i[2].shape) + " " + str(i[3].shape))
    """ 

    return to_return

"""
In a given clip, fills inbetween two frames (exclusive, I.E. the frames in between start and end),
assuming the frames in between are corrupted in some way and are not to be used so need to be filled-in by estimations.
It can only fill in a max of two frames.

"""
def fill_in_frames(clip, start, end):
    if(end-start > 3):
        raise Exception("Fill-ins for lengths of more than 2 is not yet implemented.")
    if(end-start <= 1):
        raise Exception("Must be at least one frame to fill in.")
    
    for i in range(len(clip)):
        clip[i][(start+1)] = clip[i][(start)] #Replace first frame with previous frame
        if(end-start == 3):
            clip[i][(end-1)] = clip[i][(end)] #Replace second frame with following frame if second frame is to be replaced.

    return clip

"""
Extrapolates intermediate frames by quaternion angle changes found by slerp

Process:
     -Quaternion angles are varied overtime by quaternion slerp, initial position is start, final position is end, and inbetween
      angles are extrapolated by slerp.
     -For pose face points that do not have a designated quaternion a similar process is done to other pose points except a start and
     end quaternion for them is calculated first.
     -Point positions are extrapolated as follows
           -The head position is always 0,0,0 so we just keep it that way
           -A slope for change in distance per frame between each limb/segment is found by the distance in the initial and final frame
           divided by the frame gap (rise over run)
           -The vector direction for each limb/segment is found by the quaternions for the limb segments (chain system similar to the 
           visualizor.)
           -Using the limb segment vector direction and magnitude for each frame, along with the head position for that frame, the new
           positions for each point are found for that frame.
     -Pointcloud is extrapolated as follows:
           -The colour of each point is either the same as that of the initial or final frame, depending on which is closer
           -The location of the point is oriented to the new quaternion.

"""
def fill_in_slerp(clip, start, end):
    if(end-start <= 1):
        raise Exception("Must be at least one frame to fill in.")

    pose_points_3d_list = clip[0]
    pose_angles_3d_list = clip[1]
    point_clouds_list = clip[2]
    head_angles = clip[4]
   
    times = range(0,end+1-start)
    
    """
    First, extrapolate new quaternions
    """
    for i in range(len(pose_angles_3d_list[0])): #Iterate over each quaternion that varies overtime
        slerp = Slerp([0, end-start],R.concatenate([R.from_quat(pose_angles_3d_list[start][i]),R.from_quat(pose_angles_3d_list[end][i])]))
            
        intermediate_rots = slerp(times).as_quat()
        
        i3 = 0
        for i2 in range((start),(end+1)):
            #i2 tracks the current frame, i tracks the angle, i3 tracks the timestamp in slerp
            pose_angles_3d_list[i2][i] = intermediate_rots[i3] 
            i3 = i3+1
    
    """
    Step 1.5: Calculate slerp quaternions for the facial points. 
    """
    mouth_initial_vector = pose_points_3d_list[start][9]-pose_points_3d_list[start][17] #Calculate forward vectors
    mouth_final_vector = pose_points_3d_list[end][9]-pose_points_3d_list[end][17]
    eye_initial_vector = pose_points_3d_list[start][10]-pose_points_3d_list[start][17]
    eye_final_vector = pose_points_3d_list[end][10]-pose_points_3d_list[end][17]
    
    mouth_initial = orientation_quaternion(mouth_initial_vector,absolute_upward_vector) #Calculate quaternions
    mouth_final = orientation_quaternion(mouth_final_vector,absolute_upward_vector)
    eye_initial = orientation_quaternion(eye_initial_vector,absolute_upward_vector)
    eye_final = orientation_quaternion(eye_final_vector,absolute_upward_vector)

    mouth_slerp = Slerp([0, end-start],R.concatenate([R.from_quat(mouth_initial),R.from_quat(mouth_final)])) #Calculate quaternion slerps
    eye_slerp = Slerp([0, end-start],R.concatenate([R.from_quat(eye_initial),R.from_quat(eye_final)]))
    
    mouth_quaternions = mouth_slerp(times).as_quat() #Convert to quaternion list
    eye_quaternions = eye_slerp(times).as_quat()

    """
    Second, extrapolate new points.
    
    connected pairs first two elements are initial and final point, while
    the third element and onward is the quaternion chain leading to that segment, where the final element is the segment quaternion.
    
    They are ordered by the number of segments down the head, so their initial segment is always calculated already before they
    try to use it to extrapolate the final segment.
    
    -1 signals the eye custom-calculated quaternion,
    -2 signals the mouth custom-calculated quaternion
    """
    connected_pairs = [[17, 9, -2], #Custom-calculatd quaterion, direct at head
                       [17, 10, -1], #Custom-calculatd quaterion, direct at head
                       [17, 8, 1], #quaternion 1, direct at head
                       [8, 14, 1, 2], #quaternion 2, one layer down from head
                       [8, 11, 1, 3], #quaternion 3, one layer down from head
                       [8, 7, 1, 4], #quaternion 4, one layer down from head
                       [7, 0, 1, 4, 5], #quaternion 5, two layers down from head
                       [14, 15, 1, 2, 6], #quaternion 6, two layers down from head
                       [11, 12, 1, 3, 7], #quaternion 7, two layers down from head
                       [15, 16, 1, 2, 6, 8], #quaternion 8, three layers down from head
                       [12, 13, 1, 3, 7, 9], #quaternion 9, three layers down from head
                       [0, 1, 1, 4, 5, 14], #quaternion 14, three layers down from head
                       [0, 4, 1, 4, 5, 15], #quaternion 15, three layers down from head
                       [1, 2, 1, 4, 5, 14, 12], #quaternion 12, four layers down from head
                       [4, 5, 1, 4, 5, 15, 13], #quaternion 13, four layers down from head
                       [5, 6, 1, 4, 5, 15, 13, 11], #quaternion 11, five layers down from head
                       [2, 3, 1, 4, 5, 14, 12, 10]] #quaternion 10, five layers down from head
    
    for pair in connected_pairs:
        initial_vector = pose_points_3d_list[start][pair[1]]-pose_points_3d_list[start][pair[0]] #Get vector between the pair at start frame
        final_vector = pose_points_3d_list[end][pair[1]]-pose_points_3d_list[end][pair[0]] #Get vector between the pair at end frame
        
        initial_vector_mag = np.sqrt(np.dot(initial_vector,initial_vector)) #Get magnitude of initial vector
        final_vector_mag = np.sqrt(np.dot(final_vector,final_vector)) #Get magnitude of final vector
        
        magnitude_change_overtime = (final_vector_mag-initial_vector_mag)/(end-start) #Get rate of change of magnitude.
        
        for i in range(start, (end+1)):
            rotation_matrix = np.identity(3)
            for i2 in range((len(pair)-1),1,-1): #Apply quaternions sucessivly as matrixes to get absolute in-space orientation
                if(pair[i2] > -1):
                    matrix_to_multiply = (R.from_quat(pose_angles_3d_list[i][pair[i2]])).as_matrix()
                else:
                    index = i-start
                    if pair[i2] == -1:
                        matrix_to_multiply = (R.from_quat(eye_quaternions[index])).as_matrix()
                    else:
                        matrix_to_multiply = (R.from_quat(mouth_quaternions[index])).as_matrix()
                rotation_matrix = np.matmul(matrix_to_multiply,rotation_matrix)


            rotation = R.from_matrix(rotation_matrix) #Convert absolute in-space orientation matrix to a rotation object
            
            magnitude = initial_vector_mag + (i-start)*magnitude_change_overtime #Calculate magntude for that frame.

            vector = rotation.apply(np.array([0,0,1])) * magnitude #Rotate unit vector by found quaternion and make its length magnitude.
            
            pose_points_3d_list[i][pair[1]] = pose_points_3d_list[i][pair[0]] + vector #Add vector to initial point to get final point.

    """
    Finally, fill in pointclouds and quaternions pre-flattening.
    """
    initial_rotation = (R.from_quat(head_angles[start])) #This is the initial rotation
    final_rotation = (R.from_quat(head_angles[end])) #This is the final rotation
    
    head_slerp = Slerp([0, end-start], R.concatenate([initial_rotation, final_rotation]))
    intermediate_rots = head_slerp(times).as_quat() #generate intermediate rotations of head
    
    i3 = 0
    for i in range((start+1), end):
        if abs(start-i) < abs(end-i):
            old_rotation = initial_rotation #Closer to initial rotation
            
            frame_to_target = start
        else:
            old_rotation = final_rotation #Closer to final rotation
            
            frame_to_target = end
            
        #Now calculate actual vector that would have been used in the head_rotation function.
        forward_vector = np.array([0,0,1])
        forward_vector = old_rotation.apply(forward_vector)
        forward_vector[1] = 0
        old_rotation_inverted = (R.from_quat(orientation_quaternion(forward_vector,absolute_upward_vector)))
        
        #Dont like that I have to copy here but want to make sure the initial and final frames dont get modified.
        point_clouds_list[i,:,:3] = old_rotation_inverted.apply(np.copy(point_clouds_list[frame_to_target,:,:3])) #Undo the rotation on final or initial
        

        #Add new quaternion to angles
        new_rotation = intermediate_rots[i3+1]
        i3 = i3+1
        head_angles[i] = new_rotation
        
        #Rotate points by new vertically flattened quaternion.
        new_rotation = (R.from_quat(new_rotation))
        forward_vector = np.array([0,0,1])
        forward_vector = new_rotation.apply(forward_vector)
        forward_vector[1] = 0
        new_rotation = (R.from_quat(orientation_quaternion(forward_vector,absolute_upward_vector)))
        
        point_clouds_list[i,:,:3] = new_rotation.apply(point_clouds_list[i,:,:3])

        #Apply colour of chosen frame
        point_clouds_list[i,:,3:] = point_clouds_list[frame_to_target,:,3:]
        
    clip[0] = pose_points_3d_list 
    clip[1] = pose_angles_3d_list
    clip[2] = point_clouds_list
    clip[4] = head_angles
    
    return clip
        
        



"""
removes clips in data below minimum allowable length
"""
def remove_short_clips(data, minimum_allowable_length):
    """
    search for too short clips and check for mismatches
    """
    marked_for_removal = []
    for i3 in range(len(data)):
        current_clip = data[i3]
        if not (len(current_clip[0]) == len(current_clip[1]) == len(current_clip[2])):
            raise Exception("Mismatch in clip frame amounts between " + str(len(current_clip[0])) + ", "
                            + str(len(current_clip[1])) + ", " + str(len(current_clip[2])))
        
        if (len(current_clip[0]) < minimum_allowable_length):
            marked_for_removal.append(i3)
            #print(len(current_clip[0]))
    
    """
    remove the too short clips.
    """
    marked_for_removal.reverse()
    for i4 in marked_for_removal:
        data.pop(i4)
        
    return data


"""
From the non-pixelspace 3D pose data, the 2D pixel-space pose data, and the depth data,
construct 3D pixelspace pose data for a frame 

From the 3D pose data, construct angle data for a frame
(should probally put this in a separate function which feeds into this one). 

Returns the pointcloud shifted relative to the head position a frame 
     -Ensure the 3D pixelspace pose data is also shifted relative to the head. 
     -Figure out how to determine head orientation? 
     
Returns the position of the head before it is shifted to the origin, I.E. the position relative to the camera.
"""
def pose_extract_3d(skeleton_data_2d_frame, skeleton_data_3d_frame, depth_frame, point_cloud_frame):
    
    skeleton_points_3d, depth_multipler = from_2d_get_3d(skeleton_data_3d_frame, skeleton_data_2d_frame, depth_frame)
    
    unaltered_head_position = np.copy(skeleton_points_3d[9]) #Use this for later analysis of head positions

    skeleton_angles_3d = get_3d_angles(skeleton_points_3d)
    
    point_cloud_frame[:, 2] *= depth_multipler

    skeleton_points_3d, point_cloud_frame = shift_to_head(skeleton_points_3d, point_cloud_frame)

    return skeleton_points_3d, skeleton_angles_3d, point_cloud_frame, unaltered_head_position

"""
From nonpixelspace 3d data, pixelspace 2d data, and depth data, get the
pixelspace 3d data (including the actual location of the head) for a frame. 
"""
head_square_2D_rad = 10 #How many pixels away from the original head position should it be able to move for depth correction
def from_2d_get_3d(pose_frame_3d, pose_frame_2d, depth_frame):
    pose_frame_2d[:, 1] -= 120
    pose_frame_2d[:, 1] *= -1
    pose_frame_2d[:, 1] += 120
    """
    Using depth data and 2d points, make approximated 3D points. 
    FOV spread is done on the points at the end as well, same as the point cloud. 
    """
    extrapolated_2d_points = []
    for keypoint in pose_frame_2d:
        xloc = round(keypoint[0])
        yloc = round(keypoint[1])
        try:
            depth = depth_frame[xloc][yloc].item() #get the depth at that point
            extrapolated_2d_points.append(np.array([xloc,yloc,depth])) 
        except:
            #print("The 2d pose point was out of frame!")
            extrapolated_2d_points.append(np.array([xloc,yloc,-1])) #append this to indicate that the point is out of frame, do not use
    extrapolated_2d_points = np.stack(extrapolated_2d_points) #Stack limb points into a mini pointcloud.
    
    head_x = round(extrapolated_2d_points[0][0].item())
    head_y = round(extrapolated_2d_points[0][1].item())
    lowest_depth = extrapolated_2d_points[0][2].item()
    new_x = head_x
    new_y = head_y
    for i in range(head_x-head_square_2D_rad-1, head_x+head_square_2D_rad):
        for i2 in range(head_y-head_square_2D_rad-1,head_y+head_square_2D_rad):
            distance_vector = np.array([head_x,head_y])-np.array([i,i2])
            distance_from_original = np.sqrt(np.sum(np.square(distance_vector)))
            
            if(distance_from_original<head_square_2D_rad): #Check if within specified radius of the head.
                try:
                    depth = depth_frame[i][i2].item()
                    if depth < lowest_depth: #New lowest depth value found, set new_x and new_y, head x and y will be changed to this location
                        lowest_depth = depth
                        new_x = i
                        new_y = i2
                except Exception:
                    pass
                
    extrapolated_2d_points[0][0] = new_x
    extrapolated_2d_points[0][1] = new_y
    extrapolated_2d_points[0][2] = lowest_depth
   
    """
    Using points that 2D and 3D keypoints have in common, make an average conversion factor between the
    space of the 3D points and the pointcloud space.
    Common points are:
    Left and right hand
    Left and right elbow
    Left and right shoulder
    Left and right hip
    Left and right knee
    Left and right foot
    Mouth/Nose
    For a total of 12 points.
    
    3D keypoints are in human3.6M format, 2D keypoints are in COCO format (see mmpose docs):
    
    
    In addition the common point's depth are corrected via the following procedure:
        -Get vector from just x,y points from the head from 3d pose data
        -Get vector from x,y,z points from the head from 3d pose data
        -Get vector from just x,y points from the head from 2d pose data
        -From this extrapolate vector for x,y,z points from the head from 2d pose data
        -Add this vector to the head to get a new extrapolated point, use the depth from this point as the new depth.
    This helps to correct for disprepancies in the conversion rate caused by body keypoints moving into the background.
    However, this does not help when the head is the point that is in the background, resulting in a pattern where the conversion
    rate is reasonable for frames except for when the head is in the background where it then becomes incorrect again.
    """
    common_points_2d_3d = {5:11,6:14,7:12,8:15,9:13,10:16,11:4,12:1,13:5,14:2,15:6,16:3,0:9} #keys are 2D point index, value is corresponding 3D point index.
    sum_3d = 0
    sum_2d = 0
    uncovered_points = list(common_points_2d_3d.keys())
    
    head_pos_2D = extrapolated_2d_points[0]
    #Compensate weird 3D transformations. 
    head_pos_3D = np.array([(-1*pose_frame_3d[9][0]),pose_frame_3d[9][2],pose_frame_3d[9][1]])
    for point in common_points_2d_3d:
        if(point != 0):
            tip_2D = extrapolated_2d_points[point]
            #Compensate for weird 3D transformations
            tip_3D = np.array([(-1*pose_frame_3d[common_points_2d_3d[point]][0]),pose_frame_3d[common_points_2d_3d[point]][2],pose_frame_3d[common_points_2d_3d[point]][1]])
        
            extrapolated_vector_2D = np.array([tip_2D[0], tip_2D[1]])-np.array([head_pos_2D[0],head_pos_2D[1]])
            pose_vector_2D = np.array([tip_3D[0], tip_3D[1]])-np.array([head_pos_3D[0],head_pos_3D[1]])
            pose_vector_3D = tip_3D-head_pos_3D
        
            mini_conversionfactor = np.sqrt(np.sum(np.square(extrapolated_vector_2D)))/np.sqrt(np.sum(np.square(pose_vector_2D)))
            new_extrapolated_position = (pose_vector_3D/np.sqrt(np.sum(np.square(pose_vector_3D))))*mini_conversionfactor + head_pos_2D
            extrapolated_2d_points[point] = new_extrapolated_position
    
    #extrapolated_2d_points = cloud_FOV_spread(extrapolated_2d_points, FOV, FOV, 320, 240) 

    #extrapolated_2d_points = cloud_FOV_spread(extrapolated_2d_points, FOV, FOV, 320, 240) #Do FOV spread on limb points

    for i in common_points_2d_3d: #Loop through each point
        uncovered_points.remove(i) #This point is being covered, remove it from uncovered
        for i2 in uncovered_points: #Loop through points that have not been covered yet
            if extrapolated_2d_points[i][2]!=-1 and extrapolated_2d_points[i2][2]!=-1:
                vector_3D = pose_frame_3d[common_points_2d_3d[i]]-pose_frame_3d[common_points_2d_3d[i2]]
                vector_2D = extrapolated_2d_points[i] - extrapolated_2d_points[i2]
            
                sum_3d = sum_3d + np.sqrt(np.sum(np.square(vector_3D))) #Add 3d distance to 3d sum
                sum_2d = sum_2d + np.sqrt(np.sum(np.square(vector_2D))) #Add 2d distance to 2d sum
                #print(np.sqrt(np.sum(np.square(vector_2D)))/np.sqrt(np.sum(np.square(vector_3D))))


    conversion_factor = sum_2d/sum_3d #multiply this conversion factor by 3d length to convert it to a 2d length
    #print("Conversion factor is: " + str(conversion_factor))

    """
    Use conversion factor to extrapolate pixelspace 3d points from a reference point, the conversion factor,
    and the nonpixelspace 3d points. 
    """
    head_position = extrapolated_2d_points[0] #Using the 2D mouth as the extrapolated head position
    
    #head_position = np.array([head_position])
    #head_position = cloud_FOV_spread(head_position, FOV, FOV, 320, 240) #Spread head location
    #head_position = head_position[0]
    
    #print(head_position)
    pixelspace_points_3d = []
    for i in range(len(pose_frame_3d)):
        if(i==9):
            pixelspace_points_3d.append(head_position) #9 is the head position, assign
        else: #Calculate position via relative position to mouth
            relative_position = pose_frame_3d[i] - pose_frame_3d[9] #Turn point 9 (mouth) into the origin
            
            """
            Here I apply some transformations to the 3D points while the origin is at the mouth.
            """
            z = relative_position[1]
            y = relative_position[2]
            relative_position[2] = z
            relative_position[1] = y
            relative_position[0] = -relative_position[0]
            #Convert to pixelspace and add head_position to restore origin
            """
            Put point in realspace.
            """
            absolute_position = relative_position*conversion_factor + head_position

            pixelspace_points_3d.append(absolute_position)
         
    """
    Add a derived head position as the final element.
    The approximate head position is found by taking the direction vector from the mouth (9) to in between
    the eyes (10), then making that vector the same length as the vector to the mouth, before finally adding
    this vector to the middle shoulder point (8)
    """
    direction_vector = pixelspace_points_3d[10]-pixelspace_points_3d[9] #get direction vector
    magnitude_vector = pixelspace_points_3d[9]-pixelspace_points_3d[8] #Use this to get desired magnitude
    unit_vector = direction_vector/np.sqrt(np.sum(np.square(direction_vector))) #get unit vector of direction vector
    head_vector = np.sqrt(np.sum(np.square(magnitude_vector))) * unit_vector #Use magnitude and unit vector to get final vector
    head_point = head_vector + pixelspace_points_3d[8] #add final vector to mid shoulder
    
    pixelspace_points_3d.append(head_point) #append head point as the final element (element 17)


    """
    z_dividedby_xy_ratios = [] 
    for i in pixelspace_points_3d:
        for i2 in pixelspace_points_3d:
            if not np.array_equal(i2,i):
                xy_vector = np.array([i[0],i[1]])-np.array([i2[0],i2[1]])
                xy_distance = np.sqrt(np.sum(np.square(xy_vector)))
                z_distance = abs(i[2]-i2[2])
                z_dividedby_xy_ratios.append((z_distance/xy_distance))
    average_z_xy_ratio = sum(z_dividedby_xy_ratios)/len(z_dividedby_xy_ratios)

    """
    
    """
    Stack found pixelspace 3d points and spread.
    """
    
    pixelspace_points_3d = np.stack(pixelspace_points_3d)
    
    new_pixelspace_head = cloud_FOV_spread(np.expand_dims(np.copy(pixelspace_points_3d[17]), axis=0), FOV, FOV, 320, 240) #Do FOV spread on limb points
    displacement_factor = new_pixelspace_head-pixelspace_points_3d[17]
    pixelspace_points_3d = pixelspace_points_3d+displacement_factor
    
    """
    #pixelspace_points_3d = cloud_FOV_spread(pixelspace_points_3d, FOV, FOV, 320, 240) #Do FOV spread on limb points

    xy_distances = [] 
    z_distances = []
    for i in pixelspace_points_3d:
        for i2 in pixelspace_points_3d:
            if not np.array_equal(i2,i):
                xy_vector = np.array([i[0],i[1]])-np.array([i2[0],i2[1]])
                xy_distance = np.sqrt(np.sum(np.square(xy_vector)))
                z_distance = abs(i[2]-i2[2])
                xy_distances.append((xy_distance))
                z_distances.append(z_distance)
    new_average_xy_distance = sum(xy_distances)/len(xy_distances)
    new_average_z_distance = sum(z_distances)/len(z_distances)

    new_desired_z_distance = new_average_xy_distance*average_z_xy_ratio
    new_z_multiplier = new_desired_z_distance/new_average_z_distance
    
    for i in pixelspace_points_3d:
        i[2] *= new_z_multiplier

    #
    Return pixelspace points and z multiplier.
    """
    return pixelspace_points_3d, 1#new_z_multiplier



"""
From 3d skeleton pixelspace data, get 3d angles for the joints.

In the mujoco simulation the angles are as follows:
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
Not going to actually use these angles, doing quaternion of vector change. (I.E change from head-middle shoulder vector to
middle shoulder-abdomen vector, continuing down a chain like this.). Up vectors for the quaternions are defined in segments as follows 
(easier to read version of this same definition is in discord channel).:



Definition of forward: line from the "head" position (the one we custom calculate, the last keypoint) to the mouth position. 

Up vectors are divided into 4 limb sections and 2 torso sections.

Chest area: Up vector for shoulder lines and mid-shoulder to abdomen line: Cross product of shoulder-shoulder line and midshoulder-abdomen line, flipped to point closer to forward vector.

Hip area: Up vector for the hip lines and the abdomen-midhip line are defined by the cross product of the abdomen-midhip line and the hip-hip line, flipped to point closer to the chest area up vector.


Limb sections:
Elbow area: 
     -The up vector for the bicep is calculated from the bicep vector minus the forearm vector, then made perpendicular to the bicep vector, before being flipped closer to the chest orientation if nessecary
     -The up vector for the forearm is calculated from: first take bicep_upxbicep_forward=bicep_sideways, then take take forearm_forwardxbicep_sideways to get forearm_up

Knee area: 
     -The up vector for the thigh is calculated from the thigh vector minus the calf vector, then made perpendicular to the thigh vector, before being flipped to be closer to the hip orientation if nessecary. 
     -The up vector for the calf is calculated from: first take thigh_upxthigh_forward=thigh_sideways, then take take calf_forwardxthigh_sideways to get calf_up 

     
Note that the knee and bicep calculations rely on the assumption that the upper part of the bicep and knee cannot face backwards
"""
def get_3d_angles(keypoints_3d):
    #calculate forward vector from averaging out vector to eyes and vector to mouth 
    forward_vector_1 = keypoints_3d[9]-keypoints_3d[17] #Tip-tail, mouth-head
    forward_vector_2 = keypoints_3d[10]-keypoints_3d[17] #Tip-tail, eyes-head
    forward_vector = forward_vector_1+forward_vector_2
    forward_vector = forward_vector/np.sqrt(np.dot(forward_vector,forward_vector))
    
    #calculate the chest up vector from shoulder-shoulderxmidshoulder-abdmomen
    shoulder_shoulder_line = keypoints_3d[14]-keypoints_3d[11] #Right shoulder minus left shoulder
    midshoulder_abdomen_line = keypoints_3d[7]-keypoints_3d[8] #Abdomen minus midshoulder
    chest_up_vector = -np.cross(shoulder_shoulder_line,midshoulder_abdomen_line)
    
    #calculate the hip up vector from 
    hip_hip_line = keypoints_3d[1]-keypoints_3d[4] #Right hip minus left hip
    abdomen_midhip_line = keypoints_3d[0]-keypoints_3d[7] #midhip minus abdomen
    hip_up_vector = -np.cross(hip_hip_line,abdomen_midhip_line)
    
    #Use this function to calculate upper limbs (knees or biceps)
    def upperlimb_up_vector_calc(hip_or_chest_up_vector,hand_or_foot,knee_or_elbow,hip_or_shoulder,flip_to_torso=True):
        upper_limb_vector = knee_or_elbow-hip_or_shoulder #knee/eblow minus hip/shoulder (tip minus tail)
        forelimb_vector = hand_or_foot-knee_or_elbow #foot/hand minus knee/elbow (tip minus tail)
        upper_limb_up_vector = upper_limb_vector-forelimb_vector
        
        #If flipped vector is more similar to hip/chest use it
        if flip_to_torso and (np.dot(upper_limb_up_vector,hip_or_chest_up_vector) < np.dot(-upper_limb_up_vector,hip_or_chest_up_vector)): 
            upper_limb_up_vector = -upper_limb_up_vector
            
        return upper_limb_up_vector
    
    #Call upperlimb function to calculate upper limbs (knees and biceps)
    #Biceps are not flipped to torso as biceps can face backwards, unfortunantly no real way to verify that it is not reversed.
    left_bicep_vector = upperlimb_up_vector_calc(chest_up_vector,keypoints_3d[13],keypoints_3d[12],keypoints_3d[11], False)
    right_bicep_vector = upperlimb_up_vector_calc(chest_up_vector,keypoints_3d[16],keypoints_3d[15],keypoints_3d[14], False)
    left_thigh_vector = upperlimb_up_vector_calc(hip_up_vector,keypoints_3d[6],keypoints_3d[5],keypoints_3d[4])
    right_thigh_vector = upperlimb_up_vector_calc(hip_up_vector,keypoints_3d[3],keypoints_3d[2],keypoints_3d[1])
    

    #Use this function to calculate lower limb up vectors (calf and forearm)
    def lower_limb_up_vector_calc(upper_limb_up_vector, hand_or_foot, elbow_or_knee, shoulder_or_hip):
        upper_limb_forward_vector = elbow_or_knee - shoulder_or_hip #Knee/elbow minus hip/shoulder (tip-tail)
        side_vector = np.cross(upper_limb_up_vector, upper_limb_forward_vector)
        lower_limb_forward_vector = hand_or_foot - elbow_or_knee #Hand/foot minus elbow/knee
        lower_limb_up_vector = np.cross(lower_limb_forward_vector,side_vector)

        return lower_limb_up_vector
    
    #Use function to calculate lower limb up vectors (forearm and calf)
    left_forearm_vector = lower_limb_up_vector_calc(left_bicep_vector,keypoints_3d[13],keypoints_3d[12],keypoints_3d[11])
    right_forearm_vector = lower_limb_up_vector_calc(right_bicep_vector,keypoints_3d[16],keypoints_3d[15],keypoints_3d[14]) 
    left_calf_vector = lower_limb_up_vector_calc(left_thigh_vector,keypoints_3d[6],keypoints_3d[5],keypoints_3d[4])
    right_calf_vector = lower_limb_up_vector_calc(right_thigh_vector,keypoints_3d[3],keypoints_3d[2],keypoints_3d[1])

    """
    The first list value is the initial point, the second value is the midpoint, 
    the third value is the final point for junctions.
    
    The fourth and fifth value is the initial and final up vector respectivly:
    
    Note that connected pairs function is also present in the slerp function to some extent, if you change it please
    update the slerp function as well.
    """
    connected_pairs = [[17,8,14,forward_vector,chest_up_vector], [17,8,11,forward_vector,chest_up_vector], 
                       [17,8,7,forward_vector,chest_up_vector], [8,7,0,chest_up_vector,hip_up_vector], 
                       [8,14,15,chest_up_vector,right_bicep_vector], [8,11,12,chest_up_vector,left_bicep_vector], 
                       [14,15,16,right_bicep_vector,right_forearm_vector], [11,12,13,left_bicep_vector,left_forearm_vector], 
                       [1,2,3,right_thigh_vector,right_calf_vector], [4,5,6,left_thigh_vector,left_calf_vector], 
                       [0,1,2,hip_up_vector,right_thigh_vector], [0,4,5,hip_up_vector,left_thigh_vector], 
                       [7,0,1,hip_up_vector,hip_up_vector], [7,0,4,hip_up_vector,hip_up_vector]]

    quaternion_angles = []
    
    #First append absolute head quaternion orientation in space
    upward_vector = keypoints_3d[17]-keypoints_3d[8] #goes from the the midshoulder to the head
    #Upward vector is made perpendicular in the function, causing it to go straught up
    quaternion_angles.append(orientation_quaternion(forward_vector,upward_vector))

    #Second append the quaternion of the head/midshoulder, as it is the baseline that the other quaternions extend off of.
    quaternion_angles.append(orientation_quaternion(-upward_vector,forward_vector))
    
    #Then get the other rotations.
    for i in connected_pairs:
        initial_forward = keypoints_3d[i[1]] - keypoints_3d[i[0]] #Get forward vectors
        final_forward = keypoints_3d[i[2]] - keypoints_3d[i[1]]
        final_up = np.copy(i[4])

        initial_quaternion = orientation_quaternion(initial_forward,i[3]) #Get orientation quaternions
        initial_inverted = ((R.from_quat(initial_quaternion)).inv())
        
        final_forward = initial_inverted.apply(final_forward)
        final_up = initial_inverted.apply(final_up)
        
        final_quaternion = orientation_quaternion(final_forward,final_up)
        
        initial_quaternion = np.array([0,0,0,1])

        #quaternion_angles.append(final_quaternion)
        quaternion_angles.append(quaternion_between_quaternions(initial_quaternion,final_quaternion)) #Append quaternion to go from one to the other
        
    quaternion_angles = np.stack(quaternion_angles)

    return quaternion_angles


"""
Gets the quaternion required to rotate from 1 vector to another. 
https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
"""
def quaternion_from_vectors(initial, final):
    initial_unit = initial/np.sqrt(np.sum(np.square(initial)))
    final_unit = final/np.sqrt(np.sum(np.square(final)))

    crossproduct = np.cross(initial_unit, final_unit)
    w = np.array([(1 + np.dot(initial_unit, final_unit))])

    quaternion = np.concatenate([crossproduct,w]) #Forms a quaternion xyzw.
    
    quaternion = quaternion/np.sqrt(np.sum(np.square(quaternion))) #Make sure it is a unit quaternion.
    return quaternion

"""
Use a rotation matrix approach.

forward, up, can get side as a cross product, are then able to form a rotation matrix.

we then convert this rotation matrix into a quaternion using scipy
"""
def orientation_quaternion(forward_vector, up_vector):
    #Used this to brainstorm cross product stuff https://www.geogebra.org/m/d6n6rk9N
    
    #Normalize forward vector
    forward_vector = forward_vector/np.sqrt(np.dot(forward_vector,forward_vector))
    #Calculate side vector
    side_vector = np.cross(up_vector,forward_vector)
    #Normalize side vector
    side_vector = side_vector/np.sqrt(np.dot(side_vector,side_vector))
    #Re-Calculate up vector to ensure perpendicular and normalize at the same time.
    up_vector = np.cross(forward_vector,side_vector)
    
    m = np.stack([side_vector,up_vector,forward_vector]) #this gets each vector as a row
    m = np.transpose(m) #this gets each vector as a column
    r = R.from_matrix(m)

    return r.as_quat()
    
"""
Convert to rotation matrices, get rotation in between, turn back into a quaternion.
"""
def quaternion_between_quaternions(start, end):
    start = R.from_quat(start)
    end = R.from_quat(end)
    start_undone = start.inv()
    between_matrix = np.matmul(end.as_matrix(),start_undone.as_matrix())
    return R.from_matrix(between_matrix).as_quat()   

"""
Given a pointcloud and 3d pixelspace pose data, shift all points
(including pose data points) so that the head (point 17) is at the origin for a frame.
"""
def shift_to_head(pose_data_3d, pointcloud):
    head_position = pose_data_3d[17]
    head_position_colour = np.concatenate([head_position,[0,0,0]]) #Add 0s to make it broadcastable to 6 element pointcloud pixels
    
    pointcloud = pointcloud-head_position_colour #This should subtract each point by head_position, but keep a watch to make sure.
    pose_data_3d = pose_data_3d-head_position

    return pose_data_3d, pointcloud


"""
Demonstration code:

video = "50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi"
directory_end = "run/" + video

image_directory = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted Images/" + directory_end
depth_directory = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted Depths/" + directory_end
pose_data_2d = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Pose Estimations/2D/" + directory_end + ".npy"
pose_data_3d = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Pose Estimations/3D/" + directory_end + ".npy"
array = convert_directory(image_directory, depth_directory,video)

data = process_data(pose_data_2d,pose_data_3d,depth_directory,array,video);


pointcloud_stringmade = numpy_vid_to_text(data[0][2])
points_3d_stringmade = numpy_vid_to_text(data[0][0])
angles_3d_stringmade = numpy_vid_to_text(data[0][1])

with open("pointclouds.txt", "w") as text_file:
    text_file.write(pointcloud_stringmade)
with open("points.txt", "w") as text_file:
    text_file.write(points_3d_stringmade)
with open("angles.txt", "w") as text_file:
    text_file.write(angles_3d_stringmade)
"""

"""
Copied the below codee from Video_Frame_Extractor.py and modified it. 

The below iterator is made to extract from the HMDB51 dataset's directory structre. 
The frame extraction function however works for whatever, it just spits out its output images
into whatever is set as the current directory for the program, and you can feed an absolute path
into the function as input. 
"""

def from_data_iterator(dataset_directory = "/mnt/e/ML-Training-Data/HMDB51/Dataset/"):
    covered_list_file = dataset_directory + "PointCloudsCoveredList.txt"
    #Yoinked these from google to save time.
    def add_to_covered_list(covered):
        try: 
            with open(covered_list_file, 'a') as file: 
                file.write(covered + '\n') 
        except Exception as e: 
            print(f"Error: {e}")
    def checK_covered_list(to_check): 
        with open(covered_list_file, 'r') as fp: 
             data = fp.read() 
             return to_check in data     

    inputDirectory = dataset_directory + "Dataset Extracted Images" 
    depthDirectory = dataset_directory + "Dataset Extracted Depths"
    poseData2D_Directory = dataset_directory + "Dataset Pose Estimations/2D/"
    poseData3D_Directory = dataset_directory + "Dataset Pose Estimations/3D/"

    outputDirectory = dataset_directory + "Dataset Extracted Point Clouds" 

    quality_filter_keywords = ["_med_", "_goo_"] #Requires these words be in the file in order to bother saving it


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

        skip_occured = False
        skips = 0
        videos_covered = 0
        clips_saved = 0
        for video in videos:
            print(("Covered " + str(videos_covered) + " videos, " + str(clips_saved) + " clips saved."), end='\r')
            bother_with_video=False
        
            for keyword in quality_filter_keywords: #Check for required quality keywords before bothering to make training data with it. 
                if keyword in video:
                    bother_with_video=True
                

            to_check = action_output_directory + "/" + video
            
            """
            This bit of code is usefuk to uncomment if you are redoing data collection and you want to make sure clips that are known
            to not make it past filters are discarded, run it over the old data.
            if(not os.path.exists((to_check + "_clip_0/pointcloud.npy"))):
                add_to_covered_list(to_check)
            """
            
            if(os.path.exists((to_check + "_clip_0/pointcloud.npy")) or checK_covered_list(to_check)): #Skips finished files to resume.
                skip_occured = True
                skips = skips + 1
                continue
        
            if(skip_occured):
                print("Skipped " + str(skips) + " times to " + video)
                print(("Covered " + str(videos_covered) + " videos, " + str(clips_saved) + " clips saved."), end='\r')
                skips = 0
                skip_occured = False

            if bother_with_video:
                depth_location = depthDirectory +"/" + action + "/" + video
                image_location = inputDirectory +"/" + action + "/" + video
                pose_data_2d = poseData2D_Directory +"/" + action + "/" + video + ".npy"
                pose_data_3d = poseData3D_Directory +"/" + action + "/" + video +  ".npy"
        
                try:
                    point_clouds = convert_directory(image_location,depth_location,video)
                except Exception:
                    print("Error occurred in creating pointcloud for " + action + "/" + video + ", skipping.")
                    continue #Added this because sometimes (very rare) the depth extractor does 1 less frame than the image extractor.
                data = process_data(pose_data_2d,pose_data_3d,depth_location,point_clouds,video)
        
                for i in range(len(data)):
                    target_location = action_output_directory + "/" + video + "_clip_" + str(i)

                    if(not os.path.exists(target_location)):
                        os.makedirs(target_location) #Create action output directory if it does not exist
                
                    to_save_3D_pose = data[i][0] #Do not need to save head positions segment, just first 3 elements.
                    to_save_3D_angle = data[i][1]
                    to_save_pointcloud = data[i][2]
                
                    np.save((target_location+"/skeleton_points"),to_save_3D_pose)
                    np.save((target_location+"/skeleton_angles"),to_save_3D_angle)
                    np.save((target_location+"/pointcloud"),to_save_pointcloud)
                
                    clips_saved += 1
                videos_covered += 1
            add_to_covered_list(to_check)
            
        print(action + " done...")

    print("Processing complete!")
    
def pointcloud_video_totext(inp_directory, out_directory = os.getcwd()):
    
    pointclouds = np.load((inp_directory + "/pointcloud.npy"))
    pose_points = np.load((inp_directory + "/skeleton_points.npy"))
    angles = np.load((inp_directory + "/skeleton_angles.npy"))

    pointcloud_stringmade = numpy_vid_to_text(pointclouds)
    points_3d_stringmade = numpy_vid_to_text(pose_points)
    angles_3d_stringmade = numpy_vid_to_text(angles)

    os.chdir(out_directory)
    with open("pointclouds.txt", "w") as text_file:
        text_file.write(pointcloud_stringmade)
    with open("points.txt", "w") as text_file:
        text_file.write(points_3d_stringmade)
    with open("angles.txt", "w") as text_file:
        text_file.write(angles_3d_stringmade)


"""
Can either have file create pointclouds from data, or convert those pointclouds into text files.

If first argument is to_text, it will convert the specified PointCloudVideo directory (second argument) into text
files outputted in the specified directory (third arguement). In this case 2 arguments are required or will raise an
error. Third arguement is optional as it is assumed to be current directory if unspecified.

If first arguement is anything else it will run the data iterator to convert depth, pose, and image data into
pointclouds. If the first arguement is blank it will assume the dataset is at its default location, if it is not blank
it will use the first arguement as the location for the dataset. 

"""
args = sys.argv[1:]
if(len(args)==0):
    print("Making point clouds from depth, pose, and image data")
    print()
    from_data_iterator()
    exit()
if(args[0] == "to_text"):
    if(len(args) < 2):
        raise Exception("to_text requires at least 2 additional arguements for PointCloudVideo input directory and output location.")
    else:
        if(len(args) == 2):
            pointcloud_video_totext(args[1])
        else:
            pointcloud_video_totext(args[1], args[2])
    exit()
else:
    print("Making point clouds from depth, pose, and image data")
    print()
    from_data_iterator(args[0])
    exit()