import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import plotly.graph_objects as go
import time

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def calculate_posture_score(angle, exercise_type="plank"):
    optimal_range = (130, 160) if exercise_type == "plank" else (60, 75)
    optimal_angle = sum(optimal_range) / 2
    deviation = abs(angle - optimal_angle)
    score = max(0, 100 - deviation * 2.5)
    return round(score)

def get_feedback(landmarks, exercise_type):
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    angle = calculate_angle(shoulder, hip, knee)
    if exercise_type == "plank":
        if angle > 160:
            return "Raise your hips", angle
        elif angle < 130:
            return "Lower your hips", angle
    else:
        if 60 <= angle <= 75:
            return "Good form", angle
        elif angle < 60:
            return "Raise your hips", angle
        else:
            return "Lower your hips", angle
    return "Good posture", angle

def main():
    st.title("Posture Detection (Webcam or Video)")

    mode = st.radio("Choose Input Mode", ["Webcam", "Upload Video"])
    exercise_type = st.selectbox("Exercise Type", ["plank", "mountain_climber"])

    run_detection = st.checkbox("Start Pose Detection")

    if mode == "Upload Video":
        uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    else:
        uploaded_file = None

    if run_detection:
        if mode == "Webcam":
            cap = cv2.VideoCapture(0)
        elif uploaded_file is not None:
            import tempfile, os
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            st.warning("Please upload a video file.")
            return

        stframe = st.empty()
        angles = []
        frame_id = 0

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (640, 480))
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                feedback = "No person detected"
                score = 0

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    feedback, angle = get_feedback(lm, exercise_type)
                    score = calculate_posture_score(angle, exercise_type)
                    angles.append(angle)
                    frame_id += 1

          
                cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)
                cv2.putText(frame, f"Feedback: {feedback}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)
                cv2.putText(frame, f"Score: {score}/100", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

                stframe.image(frame, channels="BGR", use_container_width=True)

                if mode == "Upload Video":
                    time.sleep(0.02)  

        cap.release()
        if uploaded_file:
            os.remove(tfile.name)

        if angles:
            st.subheader("Angle Over Time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=angles,
                x=list(range(len(angles))),
                mode='lines+markers',
                line=dict(color="royalblue"),
                name="Back Angle"
            ))
            fig.update_layout(xaxis_title="Frame", yaxis_title="Angle (degrees)",
                              title="Posture Angle Over Time", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        st.success(" Detection completed.")
if __name__ == "__main__":
    main()
