import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from PIL import Image
from scipy.signal import find_peaks, savgol_filter
import time

def extract_frame_as_image(video_path, frame_number):
    """
    비디오에서 특정 프레임을 추출하고 PIL.Image 객체로 반환합니다.

    Args:
        video_path (str): 비디오 파일 경로.
        frame_number (int): 추출할 프레임 번호.

    Returns:
        PIL.Image.Image: 추출된 프레임을 PIL 이미지 객체로 반환.
    """
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video {video_path}")

    # 원하는 프레임 번호로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # 해당 프레임 읽기
    ret, frame = cap.read()

    # 비디오 캡처 객체 해제
    cap.release()

    if not ret:
        raise ValueError(f"Error: Could not read frame {frame_number} from {video_path}")

    # 프레임을 BGR에서 RGB로 변환한 뒤 PIL.Image 객체로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    return image


def wait_until_video_loaded(video_path, max_attempts=10, delay=0.5):
    """
    비디오 파일이 로딩될 때까지 대기.

    Args:
        video_path (str): 로드하려는 비디오 파일 경로.
        max_attempts (int): 최대 시도 횟수.
        delay (float): 각 시도 간 대기 시간 (초 단위).

    Returns:
        cv2.VideoCapture: 로드된 비디오 캡처 객체.
    """
    cap = None
    for attempt in range(max_attempts):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            print(f"Debug: Video loaded successfully on attempt {attempt + 1}.")
            return cap
        else:
            print(f"Debug: Waiting for video to be ready (attempt {attempt + 1}/{max_attempts})...")
            time.sleep(delay)

    raise ValueError(f"Error: Could not open video {video_path} after {max_attempts} attempts.")

def process_video_and_smooth(video_path, exercise, window_length=31, polyorder=3):
    """
    비디오를 처리하고, 포즈 추정을 수행하며, 중간 프레임을 저장하지 않고 매끄럽게 처리된 DataFrame을 생성합니다.

    매개변수:
        video_path (str): 입력 비디오의 경로.
        exercise (str): 처리할 운동 유형.
        desired_fps (int): 처리할 프레임의 목표 FPS.
        window_length (int): Savitzky-Golay 필터의 창 길이.
        polyorder (int): Savitzky-Golay 필터의 다항식 차수.

    반환값:
        pd.DataFrame: 매끄럽게 처리된 데이터가 포함된 DataFrame 객체.
    """
    # Mediapipe Pose 초기화
    mp_pose = mp.solutions.pose
    
    # 비디오 캡처
    try:
        cap = wait_until_video_loaded(video_path)
    except ValueError as e:
        print(str(e))

    frame_interval = 1

    selected_landmarks_mapping = {
        "Side-Lateral-Raise": [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_PINKY,
            mp_pose.PoseLandmark.RIGHT_PINKY,
            mp_pose.PoseLandmark.LEFT_INDEX,
            mp_pose.PoseLandmark.RIGHT_INDEX,
            mp_pose.PoseLandmark.LEFT_THUMB,
            mp_pose.PoseLandmark.RIGHT_THUMB,
        ],
        
        "Lunge":  [
            mp_pose.PoseLandmark.NOSE,           # 0
            mp_pose.PoseLandmark.LEFT_EYE_INNER, # 1
            mp_pose.PoseLandmark.LEFT_EYE,       # 2
            mp_pose.PoseLandmark.LEFT_EYE_OUTER, # 3
            mp_pose.PoseLandmark.RIGHT_EYE_INNER,# 4
            mp_pose.PoseLandmark.RIGHT_EYE,      # 5
            mp_pose.PoseLandmark.RIGHT_EYE_OUTER,# 6
            mp_pose.PoseLandmark.LEFT_EAR,       # 7
            mp_pose.PoseLandmark.RIGHT_EAR,      # 8
            mp_pose.PoseLandmark.MOUTH_LEFT,     # 9
            mp_pose.PoseLandmark.MOUTH_RIGHT,    # 10
            mp_pose.PoseLandmark.LEFT_SHOULDER,  # 11
            mp_pose.PoseLandmark.RIGHT_SHOULDER, # 12
            mp_pose.PoseLandmark.LEFT_ELBOW,     # 13
            mp_pose.PoseLandmark.RIGHT_ELBOW,    # 14
            mp_pose.PoseLandmark.LEFT_WRIST,     # 15
            mp_pose.PoseLandmark.RIGHT_WRIST,    # 16
            mp_pose.PoseLandmark.LEFT_PINKY,     # 17
            mp_pose.PoseLandmark.RIGHT_PINKY,    # 18
            mp_pose.PoseLandmark.LEFT_INDEX,     # 19
            mp_pose.PoseLandmark.RIGHT_INDEX,    # 20
            mp_pose.PoseLandmark.LEFT_THUMB,     # 21
            mp_pose.PoseLandmark.RIGHT_THUMB,    # 22
            mp_pose.PoseLandmark.LEFT_HIP,       # 23
            mp_pose.PoseLandmark.RIGHT_HIP,      # 24
            mp_pose.PoseLandmark.LEFT_KNEE,      # 25
            mp_pose.PoseLandmark.RIGHT_KNEE,     # 26
            mp_pose.PoseLandmark.LEFT_ANKLE,     # 27
            mp_pose.PoseLandmark.RIGHT_ANKLE,    # 28
            mp_pose.PoseLandmark.LEFT_HEEL,      # 29
            mp_pose.PoseLandmark.RIGHT_HEEL,     # 30
            mp_pose.PoseLandmark.LEFT_FOOT_INDEX,# 31
            mp_pose.PoseLandmark.RIGHT_FOOT_INDEX# 32
        ]

    }

    if exercise not in selected_landmarks_mapping:
        raise ValueError(f"Exercise '{exercise}' not supported.")

    selected_landmarks = selected_landmarks_mapping[exercise]

    # DataFrame 헤더 설정
    landmarks = ['frame_no']
    for landmark in selected_landmarks:
        landmarks += [f'{landmark.name}_x', f'{landmark.name}_y', f'{landmark.name}_z']

    raw_data = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # 프레임을 RGB로 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Mediapipe Pose로 프레임 처리
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    keypoints = [frame_count]
                    for landmark_id in selected_landmarks:
                        landmark = results.pose_landmarks.landmark[landmark_id]
                        keypoints.extend([landmark.x, landmark.y, landmark.z])
                    raw_data.append(keypoints)

            frame_count += 1

    cap.release()

    # 원시 데이터를 DataFrame으로 변환
    data = pd.DataFrame(raw_data, columns=landmarks)

    # Savitzky-Golay 필터를 적용하여 매끄럽게 처리
    smoothed_data = data.copy()
    coordinate_columns = [col for col in data.columns if "_" in col and col != 'frame_no']
    for col in coordinate_columns:
        time_series = data[col].dropna().values
        if len(time_series) > window_length:
            smoothed_values = savgol_filter(time_series, window_length=window_length, polyorder=polyorder)
            smoothed_data[col] = smoothed_values

    # print(f"Smoothed data prepared as DataFrame.")

    return smoothed_data

####################################################################################################################
def detect_peaks_and_troughs(smoothed_series):
    """
    Peak Detection

    매개변수:
        smoothed_series (array-like): 매끄럽게 처리된 시간 시리즈 데이터.

    반환값:
        array: 트로프 인덱스.
    """
    # 원시 피크 및 트로프 감지
    raw_peaks, _ = find_peaks(smoothed_series, height=0, distance=5)
    raw_troughs, _ = find_peaks(-smoothed_series, distance=5)

    # 0 근처의 트로프 감지 (사용자 정의)
    zero_threshold = 0.01  # 데이터 범위에 따라 조정
    zero_troughs = np.where(np.abs(smoothed_series) < zero_threshold)[0]

    # 첫 번째 트로프가 존재할 경우 포함
    if smoothed_series[0] < smoothed_series[1]:  # 첫 지점이 트로프인지 확인
        zero_troughs = np.append(zero_troughs, 0)

    # 트로프 인덱스를 병합하고 중복 제거
    all_troughs = sorted(set(raw_troughs).union(zero_troughs))

    # 극값 감지를 정제
    refined_troughs = []
    for trough in all_troughs:
        if 0 <= trough < len(smoothed_series) - 1:  # 첫 트로프를 명시적으로 포함
            if (trough == 0 or smoothed_series[trough] < smoothed_series[trough - 1]) and \
               (trough == len(smoothed_series) - 1 or smoothed_series[trough] < smoothed_series[trough + 1]):
                refined_troughs.append(trough)

    return refined_troughs

def segment_reps(smoothed_data, reference_column="LEFT_ELBOW_y"):
    """
    피크 감지 및 세그먼트 추출을 수행하고 각 세그먼트에서 대표 프레임 5개를 반환.

    매개변수:
        smoothed_data (pd.DataFrame): 매끄럽게 처리된 데이터.
        reference_column (str): 피크 감지에 사용할 참조 열 이름.

    반환값:
        list: 각 세그먼트의 대표 프레임을 포함한 데이터프레임 리스트.
    """
    if reference_column not in smoothed_data.columns:
        raise ValueError(f"Reference column '{reference_column}' not found in the dataset.")

    reference_series = smoothed_data[reference_column].dropna().values
    reference_troughs = detect_peaks_and_troughs(reference_series)

    # 세그먼트 생성
    segments = []
    if len(reference_troughs) > 1:
        for i in range(len(reference_troughs) - 1):
            start = reference_troughs[i]
            end = reference_troughs[i + 1]

            # 세그먼트 데이터 추출
            segment = smoothed_data.iloc[start:end + 1]

            # 대표 프레임 선택: 시작, 피크, 중간점, 끝
            keypoint_columns = [col for col in segment.columns if "_" in col and col != "frame_no"]
            avg_series = segment[keypoint_columns].mean(axis=1).values
            peaks, _ = find_peaks(avg_series)

            start_idx = 0
            end_idx = len(segment) - 1
            selected_indices = [start_idx]

            if len(peaks) > 0:
                peak_idx = peaks[0]
                mid_start_peak_idx = (start_idx + peak_idx) // 2
                mid_peak_end_idx = (peak_idx + end_idx) // 2
                selected_indices.extend([mid_start_peak_idx, peak_idx, mid_peak_end_idx])
            else:
                midpoint_idx = (start_idx + end_idx) // 2
                selected_indices.append(midpoint_idx)

            selected_indices.append(end_idx)
            selected_indices = sorted(set(selected_indices))

            # 선택된 프레임 데이터 추출
            selected_data = segment.iloc[selected_indices]
            segments.append(selected_data)

            # print(f"Segment {i + 1} processed with representative frames: {selected_indices}.")
    else:
        print("추출할 세그먼트를 생성하기에 충분한 트로프가 감지되지 않았습니다.")

    return segments


####################################################################################################################
def combine_segments(segments):
    """
    두 개의 연속적인 세그먼트를 결합하여 새로운 세그먼트를 생성합니다.

    매개변수:
        segments (list): 각 세그먼트를 포함하는 데이터프레임 리스트.

    반환값:
        list: 결합된 세그먼트를 포함한 데이터프레임 리스트.
    """
    combined_segments = []
    for i in range(len(segments) - 1):
        combined_segment = pd.concat([segments[i], segments[i + 1]]).reset_index(drop=True)
        combined_segments.append(combined_segment)
        # print(f"Combined segment {i + 1} and {i + 2} into a new segment with {len(combined_segment)} rows.")

    return combined_segments



####################################################################################################################

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

def read_image_from_path(image_path):
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"Image at {image_path} not found or is empty.")
    return image

def extract_keypoints_from_image(image):
    if image is None or image.size == 0:
        raise ValueError("Image object is empty or invalid.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)
    if result.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])
    else:
        raise ValueError("Pose landmarks not detected.")

def normalize_pose(keypoints, reference_indices=(11, 12)):
    center = np.mean(keypoints[reference_indices, :2], axis=0)
    keypoints[:, :2] -= center
    ref_distance = np.linalg.norm(keypoints[reference_indices[0], :2] - keypoints[reference_indices[1], :2])
    keypoints[:, :2] /= ref_distance
    return keypoints

def draw_equal_scaled_skeleton(keypoints1, keypoints2, width, height):
    # 흰색 배경 추가
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 스켈레톤 크기 조정
    scale = min(width, height) / 4
    keypoints1[:, :2] *= scale
    keypoints2[:, :2] *= scale
    center_x, center_y = width // 2, height // 2
    keypoints1[:, :2] += [center_x, center_y]
    keypoints2[:, :2] += [center_x, center_y]

    # 관절 및 연결 디자인 개선
    for kp1, kp2 in mp_pose.POSE_CONNECTIONS:
        # 첫 번째 스켈레톤 (빨간색) - BGR 순서로 색상 설정
        cv2.line(canvas, tuple(keypoints1[kp1, :2].astype(int)), tuple(keypoints1[kp2, :2].astype(int)), (0, 255, 0), 10, lineType=cv2.LINE_AA)
        cv2.circle(canvas, tuple(keypoints1[kp1, :2].astype(int)), 15, (0, 255, 0), -1)
        cv2.circle(canvas, tuple(keypoints1[kp2, :2].astype(int)), 15, (0, 255, 0), -1)
        
        # 두 번째 스켈레톤 (초록색)
        cv2.line(canvas, tuple(keypoints2[kp1, :2].astype(int)), tuple(keypoints2[kp2, :2].astype(int)), (0, 0, 225), 10, lineType=cv2.LINE_AA)
        cv2.circle(canvas, tuple(keypoints2[kp1, :2].astype(int)), 15, (0, 0, 255), -1)
        cv2.circle(canvas, tuple(keypoints2[kp2, :2].astype(int)), 15, (0, 0, 255), -1)

    return canvas

def process_pose_comparison(image1_path, image2_object):
    try:
        # 이미지 로드 및 키포인트 추출
        image1 = read_image_from_path(image1_path)
        
        # Convert PIL.Image to NumPy array if necessary
        if isinstance(image2_object, Image.Image):  # Check if it's a PIL.Image object
            image2_object = np.array(image2_object)
        
        if image2_object is None or image2_object.size == 0:
            raise ValueError("Second image object is empty or invalid.")
        
        keypoints1 = normalize_pose(extract_keypoints_from_image(image1))
        keypoints2 = normalize_pose(extract_keypoints_from_image(image2_object))

        # 캔버스 크기 지정
        height, width, _ = image1.shape
        skeleton_image = draw_equal_scaled_skeleton(keypoints1, keypoints2, width, height)

        return skeleton_image
    except Exception as e:
        print(f"Error during pose comparison: {e}")
        return None

