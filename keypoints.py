from utils import *


# 동영상 별 폴더를 생성함  Ex) MP_data/Sad/0

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# 동영상 별 중요 프레임의 시작 위치가 저장된 파일 불러옴

with open('user.pickle', 'rb') as fr:
    user_loaded = pickle.load(fr)
    fr.close()

# open CV로 동영상들을 불러와서 KeyPorints 매핑 후 Keypoints 저장

cap = cv2.VideoCapture(make_path(v_list)) # 지정한 경로에 있는 파일 불러옴
sequence_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 동영상의 최대 길이 저장

cnt = 0

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # NEW LOOP
    # Loop through actions
    
    start = time.time() # 시작 시간 체크
    
    # 동영상 분류대로 폴더에 Keypoints 저장
    
    for action in actions:
        # Loop through sequences aka videos
        
        cnt1 = 0
        for sequence in range(no_sequences):
            
            cnt2 = 0
            
            # 만약 동영상 길이가 중요 프레임 시작부터 50프레임이 넘으면 마지막프레임의 50프레임 전부터 매핑
            if sequence_length > (user_loaded[cnt]+49) :
                cat = user_loaded[cnt]
            else :
                cat = sequence_length - 50
                
            
            
            print(make_path(v_list)) # 현재 동영상 경로 및 이름 출력
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):
                
                # Read feed
                ret, frame = cap.read()
                
                frame = imutils.resize(frame, width=1000)

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                if (frame_num < cat) :
                    cnt2 += 1
                    
                elif (frame_num > cat+49) : # 50 프레임 저장후 다음 동영상으로 변경
                    break
                    
                else : # Keypoints 를 np 파일로 저장

                    keypoints = extract_keypoints(results)
                    
                    cv2.putText(image, 'Saving ', (3,80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num - cnt2))
                    #print("Save {} : {}-{}".format(action,sequence,frame_num - cnt2))
                    np.save(npy_path, keypoints)
                
                

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
                cv2.imshow('OpenCV Feed', image) # 저장 위치 확인용 출력
                    
            
            # 동영상 경로 병경 
            if(sequence%5 == 4) :
                v_list[5] = v_list[5].replace(video_kind[sequence%5],video_kind[0])
                v_list[4] += 1
                cnt += 1 
            else :
                v_list[5] = v_list[5].replace(video_kind[sequence%5],video_kind[(sequence%5) + 1])
            
            if(v_list[3] == '_SYN' and v_list[5] == '_D.mp4') :
                v_list[3] = '_REAL'
                v_list[4] = 1
                v_list[2] += 1

            if(v_list[4] == 17) :
                v_list[4] = 1
                v_list[3] = '_SYN'    
                
            
            
            
            cnt1 += 1
            # openCV 종료 후 재 시작
            cap.release()
            cap = cv2.VideoCapture(make_path(v_list))
            sequence_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
        
    cap.release()
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()

sec = time.time()-start # 종료 시간 계산

times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]

print("실행시간 : ",times)
