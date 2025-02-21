import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize MediaPipe components
mediapipe_draw = mp.solutions.drawing_utils
mediapipe_hands = mp.solutions.hands
hand_tips = [4, 8, 12, 16, 20]

def main():
    # Set page config
    st.set_page_config(page_title="Hand Finger Counter", page_icon="✌️")
    
    # Add title and description
    st.title("Hand Finger Counter")
    st.write("Show your hand to the camera and see the finger count!")

    # Initialize the webcam
    camera = st.camera_input("Show your hand to the camera")

    # Initialize MediaPipe Hands
    with mediapipe_hands.Hands(
        min_detection_confidence=0.5,
        max_num_hands=4,
        min_tracking_confidence=0.5
    ) as hands:
        
        if camera is not None:
            # Convert the image from bytes to numpy array
            bytes_data = camera.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            
            # Process the frame and detect hands
            results = hands.process(frame)
            frame.flags.writeable = True
            
            lm_list = []
            
            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    myHands = results.multi_hand_landmarks[0]
                    for id, lm in enumerate(myHands.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append([id, cx, cy])
                    mediapipe_draw.draw_landmarks(
                        frame,
                        hand_landmark,
                        mediapipe_hands.HAND_CONNECTIONS
                    )

            # Count fingers
            fingers = []
            if len(lm_list) != 0:
                # Thumb
                if lm_list[hand_tips[0]][1] > lm_list[hand_tips[0]-1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
                # Other fingers
                for i in range(1, 5):
                    if lm_list[hand_tips[i]][2] < lm_list[hand_tips[i]-2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                total_fingers = fingers.count(1)
                
                # Draw finger count
                org = (70, 375)
                font = cv2.FONT_HERSHEY_COMPLEX
                fontScale = 6
                color = (0, 255, 0)
                thickness = 5
                
                cv2.putText(
                    frame,
                    str(total_fingers),
                    org,
                    font,
                    fontScale,
                    color,
                    thickness
                )
                
                # Display finger count in Streamlit
                st.header(f"Number of fingers shown: {total_fingers}")
            
            # Display the processed frame
            st.image(frame, channels="RGB")

if __name__ == "__main__":
    main()
