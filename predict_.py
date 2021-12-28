import re
import copy
import csv
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import json
import os
import datetime
from collections import OrderedDict
import imutils
import pickle
import time
import copy
from PIL import ImageFont, ImageDraw, Image 
import math
import glob

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from utils import *
import tensorflow as tf
import csv

# 1. New detection variables
sequence = []
sentence = []

threshold = 0.8

cnt = 0
line = 0.0
start_ = 0

with open("words.csv", "r") as f:
    reader = csv.reader(f)
    for line in reader:
        actions = line

model = tf.keras.models.load_model('1130_04_81.h5') # 저장된 모델 로드

#start = time.time()

test_label_files = sorted(glob.glob("sentences/label/*REAL01_F*"))
test_files = sorted(glob.glob("sentences/video/*"))

word_lists = []
for test_file in test_label_files:
    with open(test_file, 'r') as f:
        json_data = json.load(f)

    word_lists.append([json_data['data'][i]['attributes'][0]['name'] for i in range(len(json_data['data']))])

sentences=[]
with open("sentences.csv", "r") as f:
    reader = csv.reader(f)
    for line in reader:
        sentences = line

label_files = sorted(glob.glob("sentences/label/*"))

#frame_pos
starts = []
ends = []
for label_file in label_files:
    with open(label_file, 'r') as f:
        json_data = json.load(f)
    
    #for i in range(len(json_data['data'])):
    #    start = json_data['data'][i]['start']
    start = [json_data['data'][i]['start'] for i in range(len(json_data['data']))]
    end = [json_data['data'][i]['end'] for i in range(len(json_data['data']))]
    
    starts.append((np.array(start) * 30).astype(np.int64))
    ends.append((np.array(end) * 30).astype(np.int64))

predicts = []
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for idx, test_file in enumerate(test_files):
        true_idx = idx//80
        print("true_idx: " + str(true_idx))
        sequence = []
        sentence = []
        word_num_pos = 0
        word_num = len(starts[idx])
        matched_idx = -1
        
        cap = cv2.VideoCapture(test_file)
        frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            if ret == False:
                cap.release()
                break
            frame = imutils.resize(frame, width=1000)
            frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
        
            if frame_pos >= starts[idx][word_num_pos]:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)

                if frame_pos == ends[idx][word_num_pos]:
                    sequence = [sequence[i] if i < len(sequence) else np.zeros(1662) for i in range(50)]

                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    if res[np.argmax(res)] > threshold: # 기준치를 넘는 정확도 일경우
                            sentence.append(actions[np.argmax(res)])
                            print(actions[np.argmax(res)])
                    else:
                        sentence.append("")

                    word_num_pos += 1
                    sequence = []
                    #sentence = []
            
            if word_num_pos == word_num:
                matched_idx = -1
                matched_cnt = 10
                for i in range(len(word_lists)):
                    if len(sentence) != len(word_lists[i]):
                        continue
                    diff_cnt = len(list(set(word_lists[i]) - set(sentence)))
                    if diff_cnt < matched_cnt:
                        matched_cnt = diff_cnt
                        matched_idx = i
                            
                print(sentences[matched_idx])
                predicts.append(matched_idx == true_idx)
                print(predicts)
                cv2.rectangle(image, (0,0), (1000, 40), (245, 117, 16), -1)

                b,g,r,a = 255,255,255,0
                fontpath = "gulim.ttc"
                font = ImageFont.truetype(fontpath, 30)
                img_pil = Image.fromarray(image)
                draw = ImageDraw.Draw(img_pil)

                draw.text((3, 3),  ' '.join(sentences[matched_idx]), font=font, fill=(b,g,r,a))
                img = np.array(img_pil)


                img_name = "results/result_" + str(true_idx) + "_" + str(matched_idx) + ".png" 
                print(img_name)
                # Show to screen
                #cv2.imshow('OpenCV Feed', img)
                cv2.imwrite(img_name, img)
                word_num_pos = 0
                
                cap.release()

                break

        cap.release()
        cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()


