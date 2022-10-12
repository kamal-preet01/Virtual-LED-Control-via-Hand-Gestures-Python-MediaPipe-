import cv2
import mediapipe as mp


mediapie_draw = mp.solutions.drawing_utils
mediapie_hands = mp.solutions.hands
hand_tips=[4,8,12,16,20]


vid = cv2.VideoCapture(0)

with mediapie_hands.Hands(min_detection_confidence=0.5,max_num_hands=4,min_tracking_confidence=0.5) as hands:

    while (True):
        ret, frame = vid.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        lm_list=[]

        if results.multi_hand_landmarks:

            for hand_landmark in results.multi_hand_landmarks:
                myHands = results.multi_hand_landmarks[0]
                for id , lm in enumerate(myHands.landmark):
                    h,w,c = frame.shape
                    cx,cy = int(lm.x*w),int(lm.y*h)
                    lm_list.append([id,cx,cy])
                mediapie_draw.draw_landmarks(frame,hand_landmark,mediapie_hands.HAND_CONNECTIONS)

        fingers=[]

        if len(lm_list)!= 0:
            if lm_list[hand_tips[0]][1] > lm_list[hand_tips[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            for i in range(1,5):
                if lm_list[hand_tips[i]][2] < lm_list[hand_tips[i]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            total_no_of_fingers=fingers.count(1)
            org=(70,375)
            font=cv2.FONT_HERSHEY_COMPLEX
            fontScale=6
            color = (0, 255, 0)
            thickness=5

            if total_no_of_fingers == 0:
                cv2.putText(frame, "0" ,org,font,fontScale,color,thickness)
            elif total_no_of_fingers == 1:
                cv2.putText(frame, "1" ,org,font,fontScale,color,thickness)
            elif total_no_of_fingers == 2:
                cv2.putText(frame, "2" ,org,font,fontScale,color,thickness)
            elif total_no_of_fingers == 3:
                cv2.putText(frame, "3" ,org,font,fontScale,color,thickness)
            elif total_no_of_fingers == 4:
                cv2.putText(frame, "4" ,org,font,fontScale,color,thickness)
            elif total_no_of_fingers == 5:
                cv2.putText(frame, "5" ,org,font,fontScale,color,thickness)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

vid.release()
cv2.destroyAllWindows()
