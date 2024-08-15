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
Given the 2D pixelspace skeleton file, the 3D nonpixelspace skeleton file, the pointcloud data,
and the directory where the depth data is stored, split the data into usable training segments and process it
for training (the processing is done by pose_extract_3d, but you need to split the data first.)
"""
minimum_video_frames = 20 #discard all data clips that are below this frame length. 
fill_in_cutoff = 2 #Attempt to Fill in all gaps of this length or lower.
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
        depth_frame = (depth_directory+ "/" + video + "_frame" + str(i) + ".npy")

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
    data = []
    for i2 in range(num_frames):
        if(i2 in faulty_frames):
            previous_frame_faulty = True #Frame is faulty dont process
        else:
            pose_points_3d, pose_angles_3d, point_cloud, head_pos = pose_extract_3d(skeleton_frames_2d[i2], skeleton_frames_3d[i2],
                                                                      depth_frames[i2], point_clouds[i2]) #Process frame
            if(previous_frame_faulty): #Make a new clip with gathered frame data, previous frames were marked faulty.
                data.append([np.array([pose_points_3d]),
                             np.array([pose_angles_3d]),
                             np.array([point_cloud]),
                             np.array([head_pos])])
                previous_frame_faulty=False
            else:
                index = (len(data)-1) #Append gathered frame data to pre-existing clip.
                data[index][0] = np.concatenate([data[index][0], np.array([pose_points_3d])], axis=0)
                data[index][1] = np.concatenate([data[index][1], np.array([pose_angles_3d])], axis=0)
                data[index][2] = np.concatenate([data[index][2], np.array([point_cloud])], axis=0)
                data[index][3] = np.concatenate([data[index][3], np.array([head_pos])], axis=0)
    
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
        clips = process_clip(clip, fill_in_cutoff)
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
This function identifies issues based on changes in head position.
The main issues it identifies are:
     -Brief "islands" where the head is in a drastically different position, likely caused by being placed in background
     instead of foreground. In this case it tries to mask the issue.
     -Jumps from the head hanging out in one position to hanging out in another, likely caused by a camera angle change.
     In this case in splits the clip.
"""
z_score_cutoff = 1.75 #The z-score cutoff to determine which lengths are large enough jumps to be anomalies
def process_clip(data, fill_in_cutoff):
    head_gaps = [] #This is a list of the distances between frames
    for i in range((len(data[3])-1)):
        distance_vector = data[3][i]-data[3][(i+1)]
        distance = np.sqrt(np.sum(np.square(distance_vector)))
        head_gaps.append(distance)
    gap_indicators = []
    """
    To find the gap indicators we need to do some statistics to find large deviations from the mean,
    finding the mean, standard deviation, z score.
    """
    #Calculate mean
    sum_for_mean = 0
    for i2 in head_gaps:
        sum_for_mean = sum_for_mean+i
    mean = sum_for_mean/len(head_gaps)
    
    #Calculate standard deviation
    sum_for_deviation = 0
    for i3 in head_gaps:
        sum_for_deviation = sum_for_deviation+((i3-mean)**2)
    
    standard_deviation = (sum_for_deviation/(len(head_gaps)-1))**0.5
    
    #Calculate z scores and use to find large jumps in data.
    gap_indicators.append(-1) #First and final gaps should also be marked 
    for i4 in range(len(head_gaps)):
        z_score = (head_gaps[i4]-mean)/standard_deviation
        if(z_score > z_score_cutoff): #Large jump identified
            gap_indicators.append(i4)
    gap_indicators.append(len(head_gaps))#First and final gaps should also be marked 

    """
    Now that we have identified the frames with large gaps we can identify whether it is a brief change in position or
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
                indexes_to_fill_in.append(segment)
                
    
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
                indexes_to_fill_in.append(segment)

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
            data = fill_in_frames(data,(start_and_end[0]-1),(start_and_end[1]+1)) 

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
    
    print(contious_segment_lengths)
    for i in to_return:
        print(str(i[0].shape) + " " + str(i[1].shape) + " " + str(i[2].shape) + " " + str(i[3].shape))
        
    return to_return

"""
In a given clip, attempts to extrapolate the start and end frames (exclusive, I.E. the frames in between start and end),
assuming the frames in between are corrupted in some way and are not to be used so need to be filled-in by estimations. 
"""
def fill_in_frames(clip, start, end):
    print("TODO, In progress")
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
    
    skeleton_points_3d = from_2d_get_3d(skeleton_data_3d_frame, skeleton_data_2d_frame, depth_frame)
    
    unaltered_head_position = np.copy(skeleton_points_3d[9]) #Use this for later analysis of head positions

    skeleton_angles_3d = get_3d_angles(skeleton_points_3d)
    
    skeleton_points_3d, point_cloud_frame = shift_to_head(skeleton_points_3d, point_cloud_frame)

    return skeleton_points_3d, skeleton_angles_3d, point_cloud_frame, unaltered_head_position

"""
From nonpixelspace 3d data, pixelspace 2d data, and depth data, get the
pixelspace 3d data (including the actual location of the head) for a frame. 
"""
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
            print("The 2d pose point was out of frame!")
            extrapolated_2d_points.append(np.array([xloc,yloc,-1])) #append this to indicate that the point is out of frame, do not use
    extrapolated_2d_points = np.stack(extrapolated_2d_points) #Stack limb points into a mini pointcloud.
    #extrapolated_2d_points[:, [0,1]] = extrapolated_2d_points[:, [1,0]] #TEST
    #extrapolated_2d_points[:, [2,1]] = extrapolated_2d_points[:, [1,2]] #TEST
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
    Stack found pixelspace 3d points and spread.
    """
    pixelspace_points_3d = np.stack(pixelspace_points_3d)
    pixelspace_points_3d = cloud_FOV_spread(pixelspace_points_3d, FOV, FOV, 320, 240) #Do FOV spread on limb points
    
    """
    Return pixelspace points.
    """
    return pixelspace_points_3d



"""
From 3d skeleton pixelspace data, get 3d angles for a frame as follows:
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
middle shoulder-abdomen vector, continuing down a chain like this.)    
"""
def get_3d_angles(keypoints_3d):
   
    """
    The first list value is the initial point, the second value is the midpoint, 
    the final value is the final point for junctions.
    """
    connected_pairs = [[17,8,14], [17,8,11], [17,8,7], [8,7,0], [8,14,15], [8,11,12], [14,15,16], [11,12,13], [1,2,3], [4,5,6], 
                       [0,1,2],[0,4,5],[7,0,1],[7,0,4]]
    quaternion_angles = []
    for i in connected_pairs:
        initial = keypoints_3d[i[1]] - keypoints_3d[i[0]]
        final = keypoints_3d[i[2]] - keypoints_3d[i[1]]
        quaternion_angles.append(quaternion_from_vectors(initial, final))
    
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
    #print(quaternion)
    return quaternion

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
"""
image_directory = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted Images/run/50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi"
depth_directory = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Extracted Depths/run/50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi"
video = "50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi"
pose_data_2d = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Pose Estimations/2D/run/50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi.npy"
pose_data_3d = "/mnt/e/ML-Training-Data/HMDB51/Dataset/Dataset Pose Estimations/3D/run/50_FIRST_DATES_run_f_cm_np1_ba_med_12.avi.npy"
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