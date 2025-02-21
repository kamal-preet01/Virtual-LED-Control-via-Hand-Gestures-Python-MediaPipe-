import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time

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
    
    # Enhanced CSS for professional styling
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
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .led {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            position: relative;
            text-align: center;
            transition: all 0.3s ease;
        }
        .led-on {
            background: radial-gradient(circle at 30% 30%, #ffff00, #ffd700);
            box-shadow: 0 0 20px #ffd700;
            animation: glow 1.5s ease-in-out infinite alternate;
        }
        .led-off {
            background: #444;
            box-shadow: inset 0 0 10px #000;
        }
        @keyframes glow {
            from {
                box-shadow: 0 0 10px #ffd700;
            }
            to {
                box-shadow: 0 0 20px #ffd700;
            }
        }
        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 34px;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 34px;
        }
        .slider:before {
            position: absolute;
            content: "";
            height: 26px;
            width: 26px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        input:checked + .slider {
            background-color: #2196F3;
        }
        input:checked + .slider:before {
            transform: translateX(26px);
        }
        .stats-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        .camera-container {
            position: relative;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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

def create_camera_toggle():
    toggle_html = """
        <div style="display: flex; align-items: center; margin: 20px 0;">
            <label class="toggle-switch">
                <input type="checkbox" id="cameraToggle" checked>
                <span class="slider"></span>
            </label>
            <span style="margin-left: 10px; color: #666;">Camera Control</span>
        </div>
    """
    return toggle_html

def display_header():
    st.title("🖐️ Gesture-Controlled LED System")
    st.markdown("""
        <div style='text-align: center; color: #666; margin-bottom: 30px;'>
            Control virtual LEDs with hand gestures in real-time
        </div>
    """, unsafe_allow_html=True)

def main():
    initialize_page()
    display_header()
    
    # Initialize session states
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = True
    if 'finger_count' not in st.session_state:
        st.session_state.finger_count = 0
    if 'frame_time' not in st.session_state:
        st.session_state.frame_time = time.time()
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Camera Feed")
        # Camera toggle switch
        st.markdown(create_camera_toggle(), unsafe_allow_html=True)
        
        if st.session_state.camera_on:
            camera_placeholder = st.empty()
            camera = st.camera_input("", key="camera_input")
            
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
                    
                    # Calculate FPS
                    current_time = time.time()
                    fps = 1 / (current_time - st.session_state.frame_time)
                    st.session_state.frame_time = current_time
                    
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
                    
                    # Add FPS counter to frame
                    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display processed frame
                    camera_placeholder.image(frame, channels="RGB", use_column_width=True)
    
    # LED Display and Stats in second column
    with col2:
        st.markdown("### 💡 LED Control Panel")
        
        # Display LED visualization
        st.markdown(create_led_display(st.session_state.finger_count), unsafe_allow_html=True)
        
        # Display enhanced stats
        st.markdown(f"""
            <div class="stats-container">
                <h4 style='margin: 0; color: #333;'>System Statistics</h4>
                <hr style='margin: 10px 0;'>
                <p><strong>Active LEDs:</strong> {st.session_state.finger_count}/5</p>
                <p><strong>System Status:</strong> <span style='color: {"green" if st.session_state.camera_on else "red"};'>
                    {"Active" if st.session_state.camera_on else "Inactive"}</span></p>
                <p><strong>Response Time:</strong> <20ms</p>
                <p><strong>Camera:</strong> {"ON" if st.session_state.camera_on else "OFF"}</p>
            </div>
        """, unsafe_allow_html=True)

    # Enhanced Instructions
    st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 10px; margin-top: 30px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
            <h3 style='margin-top: 0;'>📝 How to Use</h3>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;'>
                <div>
                    <h4>Setup</h4>
                    <ol>
                        <li>Toggle camera switch to ON</li>
                        <li>Allow camera access</li>
                    </ol>
                </div>
                <div>
                    <h4>Control</h4>
                    <ol>
                        <li>Show your hand clearly</li>
                        <li>Extend fingers to control LEDs</li>
                    </ol>
                </div>
                <div>
                    <h4>Best Practices</h4>
                    <ol>
                        <li>Ensure good lighting</li>
                        <li>Keep hand within frame</li>
                    </ol>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Add auto-refresh using JavaScript
    st.markdown("""
        <script>
            function refreshPage() {
                if (document.getElementById('cameraToggle').checked) {
                    window.location.reload();
                }
                setTimeout(refreshPage, 100);
            }
            setTimeout(refreshPage, 100);
        </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
