
#########################################
########## Shark Detector v2 ############
#########################################
################ VIDEO ##################
# Code by: Jeremy Jenrette
# email: jjeremy1@vt.edu
# GitHub: JeremyFJ
# Date: 6/12/2023
##############################################################################################
import numpy as np
import os
import tensorflow as tf
import cv2 as cv2
import shutil
import pickle
import math
import sys
import pandas as pd
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import visualization_utils as vis_util
from PIL import Image
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


##############################################################################################
# Old frames will be erased when running this script -- SAVE YOUR DETECTIONS
data = {'video':[], 'img_name':[], 'time_s':[], 'genus': [], 'species': [], 
        'detection_threshold':[]}
dat = pd.DataFrame(data)
##############################################################################################
# Load a (frozen) Tensorflow model into memory.
SL_model="SL_modelv3"
detect_fn=tf.saved_model.load(SL_model)
specmod = dict() # load empty species model dictionary
models = "./models/"
# Loading label map
shark_class_id = 1
category_index = {shark_class_id: {'id': shark_class_id, 'name': 'shark'}}
##############################################################################################
def img_to_classify(det_img, size, batch_size):
    img_1 = img_to_array(det_img.resize(size))
    img_1 = img_1.reshape((batch_size, img_1.shape[0], 
                        img_1.shape[1], img_1.shape[2]))
    img_1 = img_1/255
    return img_1

def open_vid(videofile):
    try:
        vidcap = cv2.VideoCapture(vid_dir+videofile)
    except (FileNotFoundError):
        print('no videos found -- add them to www/video/')
        sys.exit(2)
    fps=vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(videofile)
    print("\nfps: " + str(math.ceil(fps)))
    print("total frames: " + str(frame_count))
    print("duration (s): " + str(math.ceil(duration)))
    return vidcap, fps

def retrieve_frame(vidcap, count, frame_cap):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*frame_cap)) 
    success,img = vidcap.read()
    return success,img

def detect(image_np, threshold):
    input_tensor=tf.convert_to_tensor(image_np)
    input_tensor=input_tensor[tf.newaxis, ...]
    detections=detect_fn(input_tensor)
    num_detections=int(detections.pop('num_detections'))
    detections={key:value[0,:num_detections].numpy()
            for key,value in detections.items()}
    scores = detections['detection_scores']
    conf = scores[0]
    random_image = 'none'
    txtloc = 'none'
    # Each box represents a part of the image where a particular object was detected.
    boxes = detections['detection_boxes']
    classes = detections['detection_classes'].astype(np.int64)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1,
        min_score_thresh=threshold,
        line_thickness=10,
        agnostic_mode=False)
    
    bbox = None
    if conf > threshold:
        ymin = boxes[0,0]
        xmin = boxes[0,1]
        ymax = boxes[0,2]
        xmax = boxes[0,3]
        (im_height, im_width, channel) = image_np.shape
        bbox = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        (xminn, xmaxx, yminn, ymaxx)  = bbox
        cropped_image = tf.image.crop_to_bounding_box(image_np, int(yminn), int(xminn), int(ymaxx - yminn), int(xmaxx - xminn))
        img = cropped_image.numpy()
        random_image = Image.fromarray(img)
        txtloc = (int(xmaxx)-int(0.5*(xmaxx-xminn)), int(yminn)-20)
    return image_np, conf, random_image, txtloc, bbox
##############################################################################################
# Choose your directory where the BRUVs are stored, or load your videos into './www/video/' 
# This script will iterate through each video in the specified directory 
# Use an absolute path if possible
vid_dir = './www/video/'
# vid = 'test.mp4'
# video_name = vid.split(".")[0]

# remove for loop if you want to process only one specific video
for vid in os.listdir(vid_dir): # iterate through each video in vid_dir
    if (vid.split(".")[1].lower() != ("mp4" or "mov") ): # only process with MP4 or MOV video files
        continue
    video_name = vid.split(".")[0]
    # Playing video from file
    cap, fps = open_vid(vid)
    frame_path = "./detvid/"+video_name+"_detections/"
    try:
        shutil.rmtree(frame_path)
    except (FileNotFoundError, FileExistsError):
        pass
    os.makedirs(frame_path)
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    # Define the codec and create VideoWriter object - this to write and save a detection box video
    # WARNING: Writing a detection box video consumes a lot of computational energy

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    FILE_OUTPUT = frame_path + "detection_box.avi"
    if os.path.isfile(vid_dir + FILE_OUTPUT):
        os.remove(vid_dir + FILE_OUTPUT)
    out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc('M','J','P','G'),
                          30, (frame_width, frame_height))

    ret = True
    count = 1
    interval_frame = 1
    frame_cap = 60 # grabs 15 frames per second -- at 30 fps
    # for large videos
    # mb_size = (os.path.getsize(vid_dir+vid)) * 10 ** -6 
    # if mb_size>250: # checks if video is larger than 250mb
    #     frame_cap = 100

    while ret:
        # Capture frame-by-frame
        ret, frame = retrieve_frame(cap, count, frame_cap)
        time = int(math.floor((count*(frame_cap/fps))/fps))
        if ret == True:
            thresh = 0.80 # threshold for SL to detect a shark -- adjust this based on sensitivity
            frame, conf, cropped_image, txtloc, bounding_box = detect(frame, thresh)
            if conf>thresh:
                # We have a shark!
                name = frame_path + "frame%d.jpg"%count
                img_1 = img_to_classify(cropped_image, (224,224), 1)
                frame_df = pd.DataFrame([[video_name, "frame%d"%count, time, "shark", "", conf]], 
                            columns=list(dat.columns))
                
                # Write classification to video frame and sreadsheet
                frame = cv2.putText(frame, 'shark', txtloc, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, 
                        (57,255,20), 3, cv2.LINE_AA, False)
                dat = pd.concat([frame_df, dat])
                # cv2.imwrite(name, frame) also save the frame
                cv2.imwrite("live.jpg", retrieve_frame(cap, count, frame_cap)[1]) # view unaltered frames
            out.write(frame) # for writing a detection box video 
            count = count + interval_frame
            print('frame ' + str(count), end='\r')
        else:
            break

    # When everything done, release the video capture and video write objects and save spreadsheet
    cap.release()
    dat = dat.iloc[::-1]
    dat.to_csv("./data/" + video_name + "_SDvid.csv", index=False)
    out.release() # close and save the detection box video