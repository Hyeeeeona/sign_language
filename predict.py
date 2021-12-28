from utils import *

# 동영상 경로 초기화 및 action 한글화

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['고민', '뻔뻔', '수어','남아','눈',
                   '독신','음료수','발가락','슬프다','자극'])

# 85 videos worth of data
no_sequences = 85

# Videos are going to be 50 frames in length
sequence_length = 50

video_path = 'data/'
video_name1 = 'NIA_SL_WORD'
video_num = 1
video_name2 = '_REAL'
video_num2 = 1

video_kind = ['_D.mp4','_F.mp4','_L.mp4','_R.mp4','_U.mp4']

v_list = [video_path, video_name1, video_num, video_name2, video_num2, video_kind[0]]


print(make_path(v_list))
print(v_list)

# 1. New detection variables
sequence = []
sentence = []

threshold = 0.8
cnt = 0
line = 0.0
start = 0


cap = cv2.VideoCapture(0) # OpenCv 웹캠으로 실행
start = time.time()

# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=1000)

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        #print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 오른손 Keypoint만 저장
        right = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

        # 처음 위치를 잡기 위한 시간 체크
        sec = time.time()-start

        # 위치 설정 후 동작 시작
        if start == 1:

            #print('starting')

            # Key points 추출
            keypoints = extract_keypoints(results)

            # 모델에 넣기 위해 50프레임만큼 채움

            sequence.append(keypoints)
            sequence = sequence[-50:]


            # 50프레임이 가득 찰 경우
            if len(sequence) == 50:
                # 모델에 넣어서 예측값 얻어옴
                res = model.predict(np.expand_dims(sequence, axis=0))[0]

                if res[np.argmax(res)] > threshold: # 기준치를 넘는 정확도 일경우

                    if len(sentence) > 0: # 단어가 하나라도 출력되었을 경우

                        # 중복으로 나오는 단어 무시
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else: # 단어가 하나도 출력이 안된 경우 그냥 출력
                        sentence.append(actions[np.argmax(res)])

                # 단어 일정 이상 출력 시 화면 출력을 위해서 마지막 10개만 남김
                if len(sentence) > 10:
                    sentence = sentence[-10:]


                # Viz probabilities

            #image = prob_viz(res, actions, image, colors)

        # 레이아웃 설정

        cv2.rectangle(image, (0,0), (1000, 40), (245, 117, 16), -1)

        b,g,r,a = 255,255,255,0
        fontpath = "fonts/gulim.ttc"
        font = ImageFont.truetype(fontpath, 30)
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)


        # 오른손 위치에 따라서 시작 및 종료 감지
        if sec > 10 :

            if right[1] == 0 :
                draw.text((3, 120),  'None', font=font, fill=(0,0,0,a))

            elif right[1] < line-0.1 :
                draw.text((3, 120),  'START', font=font, fill=(0,0,0,a))
                start = 1
                #print('stop')
            else :
                draw.text((3, 120),  'STOP', font=font, fill=(0,0,0,a))
                if len(sequence) == 50:
                    start = 0
                    sequence = []
                    print('stop')
        else :
            # 처음 시작시 오른손의 기준점 위치 설정
            draw.text((3, 120),  '화면에 위치를 잡아주십시오', font=font, fill=(0,0,0,a))
            line = right[1]

            #print('기준점 : ', line)


        draw.text((3, 3),  ' '.join(sentence), font=font, fill=(b,g,r,a))
        img = np.array(img_pil)

        cv2.putText(img, 'frame : ' + str(cnt), (3,80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', img)
        cnt += 1


        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()


