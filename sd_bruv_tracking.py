
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
import torch
import sys
import pandas as pd
from base64 import b64encode
from keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from models.deep_sort_pytorch.utils.parser import get_config
from models.deep_sort_pytorch.deep_sort import DeepSort

##############################################################################################
# Global Variables
CLASS_NAMES = {1: 'shark'}

# Hyperparameters
SHARK_LOCATOR_THRESHOLD = 0.80
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


def detect(image_np):
    """
    Input:
        image_np: Numpy array of frame in video
        threshold: threshold for detection
    """
    input_tensor=tf.convert_to_tensor(image_np)
    # Add batch dimension at the beginning with value None to achieve dimensions: [None, height, width, channels]
    input_tensor=input_tensor[tf.newaxis, ...]
    # Call Shark Locator model: Faster-R-CNN with 89% accuracy. Return:
    # num_detections: number of detections in the image
    # detection_boxes: coordinates of the detections in the image
    # detection_scores: confidence scores of the detections in the image
    # detection_classes: class index of the detections in the image
    detections=detect_fn(input_tensor)

    num_detections=int(detections.pop('num_detections'))

    # Take output corresponding to the first image in a batch (since model is batched
    # but we only passed one image)
    detections={key:value[0,:num_detections].numpy()
            for key,value in detections.items()}
    
    # Each box represents a part of the image where a particular object was detected.
    boxes = detections['detection_boxes'] # they are normalised
    classes = detections['detection_classes'].astype(np.int64)
    scores = detections['detection_scores']
    confidences = np.ceil(scores*100)/100

    return boxes, classes, confidences


def initialise_deepsort():
    cfg_deep = get_config()
    cfg_deep.merge_from_file("./models/deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(
        cfg_deep.DEEPSORT.REID_CKPT,
        max_dist=cfg_deep.DEEPSORT.MAX_DIST, # should be quite low as sharks move slowly
        min_confidence=SHARK_LOCATOR_THRESHOLD,
        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, # this should be higher, since sharks can easily overlap
        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE, # This should be high to make the algorithm more flexible
        max_age=cfg_deep.DEEPSORT.MAX_AGE,
        n_init=cfg_deep.DEEPSORT.N_INIT, # we might want this lower as shark detections are sparse
        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
        use_cuda=True
    )
    return deepsort



def track_sharks(deepsort, frame, boxes, classes, confidences):
    xywh_boxes = []
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    for yxyx_box in boxes:
        y_min, x_min, y_max, x_max = yxyx_box[0], yxyx_box[1], yxyx_box[2], yxyx_box[3]
        # convert to pixel unit (expected by DeepSORT)
        y_min, x_min, y_max, x_max = int(y_min * frame_height), int(x_min * frame_width), int(y_max * frame_height), int(x_max * frame_width)
        # DeepSort requires xc, yc, w, h format for bbox
        xc, yc = (x_min + x_max) / 2, (y_min + y_max) / 2
        w, h = abs(x_max - x_min), abs(y_max - y_min)
        xywh_boxes.append([xc, yc, w, h])

    # Convert to torch tensor
    xywh_boxes = torch.Tensor(xywh_boxes)
    confs = torch.Tensor(confidences)
    oids = torch.Tensor(classes)

    # Pass detections to deepsort
    outputs = deepsort.update(xywh_boxes, confs, oids, frame)
    if len(outputs) > 0:
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -1]
        cls = outputs[:, -2]
        frame = draw_boxes(frame, bbox_xyxy, identities, cls, CLASS_NAMES)
    
    return frame



def draw_boxes(frame, bbox, identities=None, categories=None, names=None, offset=0):
    """
    Input:
        frame: {np.array} image frame
        bbox: {np.array} bounding box coordinates (x1, y1, x2, y2)
        identities: {np.array} identity labels from object tracking
        categories: {np.array} category labels from object detection
        names: {list} list of class names
        offset: {int} offset to apply to bounding box coordinates (padding)
    """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset
        x2 += offset
        y1 += offset
        y2 += offset
        # cat = int(categories[i]) if categories is not None else 0
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0

        # Draw bounding box
        rectangle_color = (0, 255, 0) # should be dynamic per class
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=rectangle_color, thickness=2, lineType=cv2.LINE_AA)
        # Draw text
        # label = str(id) + ":" + names[cat]
        label = str(id) + ":" + 'shark'
        (w,h), _ = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1/2, thickness=1)
        c2=x1+w, y1-h-3
        # Draw other rectangle including bounding box
        cv2.rectangle(frame, (x1, y1), c2, color=rectangle_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(frame, str(label), (x1, y1-2), 0, 1/2, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    return frame


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

    deepsort = initialise_deepsort()

    while ret:
        # Capture frame-by-frame
        ret, frame = retrieve_frame(cap, count, frame_cap)
        time = int(math.floor((count*(frame_cap/fps))/fps))

        if ret == True:
            boxes, classes, confidences = detect(frame)
            frame = track_sharks(deepsort, frame, boxes, classes, confidences)

            # Step3: classify sharks (extract the box and classify the new sharks)
            
                
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