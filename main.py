import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_javascript import st_javascript
from PIL import Image

# Initialize MediaPipe components
mediapipe_draw = mp.solutions.drawing_utils
mediapipe_hands = mp.solutions.hands
hand_tips = [4, 8, 12, 16, 20]

def initialize_page():
    st.set_page_config(
        page_title="Smart LED Control System",
        page_icon="💡",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

def display_header():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🖐️ Gesture-Controlled LED System")
        st.markdown("""
            <div style='text-align: center; color: #666;'>
                Control virtual LEDs with hand gestures in real-time
            </div>
        """, unsafe_allow_html=True)

def main():
    initialize_page()
    display_header()
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Camera Feed")
        camera = st.camera_input("", key="camera_input")
    
    # Initialize session state for finger count if not exists
    if 'finger_count' not in st.session_state:
        st.session_state.finger_count = 0
    
    # Initialize MediaPipe Hands
    with mediapipe_hands.Hands(
        min_detection_confidence=0.7,
        max_num_hands=1,
        min_tracking_confidence=0.7
    ) as hands:
        
        if camera is not None:
            # Process image
            bytes_data = camera.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = hands.process(frame)
            frame.flags.writeable = True
            
            lm_list = []
            
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
                        mediapipe_hands.HAND_CONNECTIONS,
                        mediapipe_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mediapipe_draw.DrawingSpec(color=(0, 128, 0), thickness=2)
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
                
                st.session_state.finger_count = fingers.count(1)
            
            # Display processed frame
            st.image(frame, channels="RGB", use_column_width=True)
    
    # LED Display in second column
    with col2:
        st.markdown("### 💡 LED Status")
        
        # Use the React component for LED display
        from streamlit_elements import elements
        
        with elements("led_display"):
            from streamlit_elements import mui
            LEDDisplay(activeCount=st.session_state.finger_count)
        
        # Display stats
        st.markdown(f"""
            <div style='background-color: white; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                <h4 style='margin: 0; color: #333;'>System Statistics</h4>
                <hr style='margin: 10px 0;'>
                <p>Active LEDs: {st.session_state.finger_count}</p>
                <p>System Status: Active</p>
                <p>Response Time: <20ms</p>
            </div>
        """, unsafe_allow_html=True)

    # Instructions
    st.markdown("### 📝 Instructions")
    st.markdown("""
        1. Allow camera access when prompted
        2. Show your hand to the camera
        3. Extend or fold your fingers to control the LEDs
        4. Number of extended fingers determines number of active LEDs
        5. For best results, ensure good lighting and keep your hand within frame
    """)

if __name__ == "__main__":
    main()
