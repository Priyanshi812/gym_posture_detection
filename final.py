import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

# MediaPipe setup
mp_pose = mp.solutions.pose
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle
def calculate_posture_score(angle, exercise_type="plank"):
    if exercise_type == "plank":
        optimal_range = (130, 160)
    elif exercise_type == "mountain_climber":
        optimal_range = (60, 75)
    else:
        return 0

    optimal_angle = sum(optimal_range) / 2
    deviation = abs(angle - optimal_angle)
    factor = 2.5  # Penalty factor
    score = max(0, 100 - deviation * factor)
    return round(score)

def get_plank_feedback(landmarks):
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

    angle = calculate_angle(shoulder, hip, knee)
    if angle > 160:
        return "Raise your hips", angle
    elif angle < 130:
        return "Lower your hips", angle
    else:
        return "Good posture", angle

def get_mountain_climber_feedback(landmarks):
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

    angle = calculate_angle(shoulder, hip, knee)
    if 60 <= angle <= 75:
        return "Good form", angle
    elif angle < 60:
        return "Raise your hips", angle
    else:
        return "Lower your hips", angle

def main():
    st.title("Exercise Posture Detection")

    exercise_type = st.selectbox("Choose the exercise:", ["plank", "mountain_climber"])
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        stframe = st.empty()
        cap = cv2.VideoCapture(video_path)

        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
                    landmarks = results.pose_landmarks.landmark
                    if exercise_type == "plank":
                        feedback, angle = get_plank_feedback(landmarks)
                    else:
                        feedback, angle = get_mountain_climber_feedback(landmarks)
                    score = calculate_posture_score(angle, exercise_type)

                # Draw UI
                cv2.rectangle(frame, (0, 0), (640, 100), (0, 0, 0), -1)
                cv2.putText(frame, f"Feedback: {feedback}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Posture Score: {score}/100", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()
        os.remove(video_path)
        st.success("✅ Video analysis complete.")

if __name__ == "__main__":
    main()