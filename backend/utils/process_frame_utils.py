import cv2
import mediapipe as mp

#initialize mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Placeholder function for processing a frame with a model
def process_frame_with_model(frame):
    # Placeholder: Perform some processing on the frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # 프레임에 랜드마크를 렌더링
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=7),  # 점
                                  mp_drawing.DrawingSpec(thickness=15, color=(0, 0, 255)))

    return frame