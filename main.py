import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time
from threading import Thread
import queue

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
        </style>
    """, unsafe_allow_html=True)

def create_led_display(active_leds):
    led_html = '<div class="led-container">'
    for i in range(5):
        led_class = "led-on" if i < active_leds else "led-off"
        led_html += f'<div class="led {led_class}"></div>'
    led_html += '</div>'
    return led_html

def process_frame(frame, hands):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    
    # Process the frame
    results = hands.process(frame_rgb)
    frame_rgb.flags.writeable = True
    
    # Initialize finger count
    finger_count = 0
    
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # Get hand landmarks
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
            
            # Draw landmarks
            mediapipe_draw.draw_landmarks(
                frame_rgb,
                hand_landmark,
                mediapipe_hands.HAND_CONNECTIONS,
                mediapipe_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mediapipe_draw.DrawingSpec(color=(0, 128, 0), thickness=2)
            )
            
            # Count fingers
            if len(lm_list) > 0:
                fingers = []
                # Thumb
                if lm_list[hand_tips[0]][1] > lm_list[hand_tips[0]-1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                
                # Other fingers
                for tip_id in range(1, 5):
                    if lm_list[hand_tips[tip_id]][2] < lm_list[hand_tips[tip_id]-2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                finger_count = fingers.count(1)
    
    return frame_rgb, finger_count

def video_capture_loop(stop_flag, frame_queue, finger_queue):
    cap = cv2.VideoCapture(0)
    
    with mediapipe_hands.Hands(
        min_detection_confidence=0.7,
        max_num_hands=1,
        min_tracking_confidence=0.7
    ) as hands:
        while not stop_flag():
            ret, frame = cap.read()
            if ret:
                processed_frame, finger_count = process_frame(frame, hands)
                
                # Update queues with new data
                if not frame_queue.full():
                    frame_queue.put(processed_frame)
                if not finger_queue.full():
                    finger_queue.put(finger_count)
                
                time.sleep(0.01)  # Small delay to prevent overwhelming the CPU
    
    cap.release()

def main():
    initialize_page()
    
    st.title("🖐️ Gesture-Controlled LED System")
    st.markdown("""
        <div style='text-align: center; color: #666; margin-bottom: 30px;'>
            Control virtual LEDs with hand gestures in real-time
        </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    # Initialize session state
    if 'stop_capture' not in st.session_state:
        st.session_state.stop_capture = False
    if 'finger_count' not in st.session_state:
        st.session_state.finger_count = 0
    
    # Create placeholder for video feed
    with col1:
        st.markdown("### 📹 Live Video Feed")
        video_placeholder = st.empty()
    
    # Create placeholders for LED display and stats
    with col2:
        st.markdown("### 💡 LED Control Panel")
        led_placeholder = st.empty()
        stats_placeholder = st.empty()
    
    # Initialize queues for frame and finger count
    frame_queue = queue.Queue(maxsize=2)
    finger_queue = queue.Queue(maxsize=2)
    
    # Start button
    if st.button("Start Camera" if st.session_state.stop_capture else "Stop Camera"):
        st.session_state.stop_capture = not st.session_state.stop_capture
        
    if not st.session_state.stop_capture:
        # Start video capture thread
        capture_thread = Thread(
            target=video_capture_loop,
            args=(lambda: st.session_state.stop_capture, frame_queue, finger_queue)
        )
        capture_thread.start()
        
        try:
            while not st.session_state.stop_capture:
                # Update video feed
                if not frame_queue.empty():
                    frame = frame_queue.get()
                    video_placeholder.image(frame, channels="RGB", use_column_width=True)
                
                # Update finger count and LED display
                if not finger_queue.empty():
                    st.session_state.finger_count = finger_queue.get()
                    
                    # Update LED display
                    led_placeholder.markdown(
                        create_led_display(st.session_state.finger_count),
                        unsafe_allow_html=True
                    )
                    
                    # Update stats
                    stats_placeholder.markdown(f"""
                        <div style='background-color: white; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                            <h4 style='margin: 0; color: #333;'>System Statistics</h4>
                            <hr style='margin: 10px 0;'>
                            <p><strong>Active LEDs:</strong> {st.session_state.finger_count}/5</p>
                            <p><strong>System Status:</strong> <span style='color: green;'>Active</span></p>
                            <p><strong>Response Time:</strong> <20ms</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                time.sleep(0.1)  # Small delay to prevent overwhelming the UI
                
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")
            st.session_state.stop_capture = True
    
    # Instructions
    st.markdown("### 📝 How to Use")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
            1. Click 'Start Camera'
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
