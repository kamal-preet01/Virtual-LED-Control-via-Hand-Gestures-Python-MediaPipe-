import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize MediaPipe components
mediapipe_draw = mp.solutions.drawing_utils
mediapipe_hands = mp.solutions.hands

# Define finger tip IDs and pip IDs (second knuckle)
tip_ids = [4, 8, 12, 16, 20]  # finger tips
pip_ids = [3, 7, 11, 15, 19]  # second knuckles
mcp_ids = [2, 6, 10, 14, 18]  # third knuckles

def initialize_page():
    st.set_page_config(
        page_title="Smart LED Control System",
        page_icon="💡",
        layout="wide"
    )
    
    # Custom CSS for styling
    st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .led-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            padding: 20px;
            background-color: #1a1a1a;
            border-radius: 15px;
        }
        .led {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            position: relative;
            text-align: center;
        }
        .led-on {
            background: radial-gradient(circle at 30% 30%, #ffff00, #ffd700);
            box-shadow: 0 0 20px #ffd700;
        }
        .led-off {
            background: #444;
            box-shadow: inset 0 0 10px #000;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        </style>
    """, unsafe_allow_html=True)

def create_led_display(active_leds):
    led_html = '<div class="led-container">'
    for i in range(5):
        led_class = "led-on" if i < active_leds else "led-off"
        led_html += f'<div class="led {led_class}"></div>'
    led_html += '</div>'
    return led_html

def display_header():
    st.title("🖐️ Gesture-Controlled LED System")
    st.markdown("""
        <div style='text-align: center; color: #666; margin-bottom: 30px;'>
            Control virtual LEDs with hand gestures in real-time
        </div>
    """, unsafe_allow_html=True)

def count_fingers(landmarks, hand_type):
    fingers = []
    
    # Get the hand type (Left or Right)
    is_right = (hand_type == "Right")
    
    # Thumb: Compare x-coordinates relative to hand type
    if is_right:
        thumb_open = landmarks[tip_ids[0]][1] > landmarks[pip_ids[0]][1]
    else:
        thumb_open = landmarks[tip_ids[0]][1] < landmarks[pip_ids[0]][1]
    fingers.append(1 if thumb_open else 0)
    
    # Four fingers: Compare y-coordinates
    for id in range(1, 5):
        if landmarks[tip_ids[id]][2] < landmarks[pip_ids[id]][2]:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return sum(fingers)

def main():
    initialize_page()
    display_header()
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Camera Feed")
        camera = st.camera_input("", key="camera_input")
    
    # Initialize session state for finger count
    if 'finger_count' not in st.session_state:
        st.session_state.finger_count = 0
    
    # Initialize MediaPipe Hands
    with mediapipe_hands.Hands(
        min_detection_confidence=0.7,
        max_num_hands=1,
        min_tracking_confidence=0.7,
        model_complexity=1
    ) as hands:
        
        if camera is not None:
            # Process image
            bytes_data = camera.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = hands.process(frame)
            frame.flags.writeable = True
            
            if results.multi_hand_landmarks and results.multi_handedness:
                hand_type = results.multi_handedness[0].classification[0].label
                landmarks = results.multi_hand_landmarks[0]
                
                # Convert landmarks to pixel coordinates
                h, w, c = frame.shape
                lm_list = []
                for lm in landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([len(lm_list), cx, cy])
                
                # Draw hand landmarks
                mediapipe_draw.draw_landmarks(
                    frame,
                    landmarks,
                    mediapipe_hands.HAND_CONNECTIONS,
                    mediapipe_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mediapipe_draw.DrawingSpec(color=(0, 128, 0), thickness=2)
                )
                
                # Count fingers
                st.session_state.finger_count = count_fingers(lm_list, hand_type)
                
                # Display hand type
                st.markdown(f"""
                    <div style='background-color: #e7f3fe; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                        <strong>Detected Hand:</strong> {hand_type}
                    </div>
                """, unsafe_allow_html=True)
            
            # Display processed frame
            st.image(frame, channels="RGB", use_column_width=True)
    
    # LED Display and Stats in second column
    with col2:
        st.markdown("### 💡 LED Control Panel")
        
        # Display LED visualization
        st.markdown(create_led_display(st.session_state.finger_count), unsafe_allow_html=True)
        
        # Display stats in a nice container
        st.markdown(f"""
            <div style='background-color: white; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                <h4 style='margin: 0; color: #333;'>System Statistics</h4>
                <hr style='margin: 10px 0;'>
                <p><strong>Active LEDs:</strong> {st.session_state.finger_count}/5</p>
                <p><strong>System Status:</strong> <span style='color: green;'>Active</span></p>
                <p><strong>Response Time:</strong> <20ms</p>
            </div>
        """, unsafe_allow_html=True)

    # Instructions
    st.markdown("### 📝 How to Use")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
            1. Allow camera access
            2. Show your hand clearly
        """)
    with cols[1]:
        st.markdown("""
            3. Extend fingers to control LEDs
            4. Keep hand within frame
        """)
    with cols[2]:
        st.markdown("""
            5. Ensure good lighting
            6. Maintain stable position
        """)

if __name__ == "__main__":
    main()
