import streamlit as st
from PIL import Image
from utils.process_frame_utils import process_frame_with_model
from utils.report_utils import *
import tempfile

# 상태 전환 및 페이지 이동을 위한 콜백 함수
def toggle_camera():
    st.session_state.camera_active = not st.session_state.camera_active
    # 카메라가 꺼질 때 페이지를 2로 전환
    if not st.session_state.camera_active and st.session_state.video_path:
        st.session_state.page = 2

def click_diagnosis():
    if "video_path" in st.session_state and st.session_state.video_path:
        st.session_state.page = 2

def first_page():
    st.write(" ")
    st.write(" ")
    st.header("🏋️‍♂️ 운동 선택하기")

    if "exercise" not in st.session_state:
        st.session_state.exercise = "Side-Lateral-Raise"

    # 운동 선택
    exercise = st.selectbox("운동 종류 선택:", ["Side-Lateral-Raise", "Push-Up", "Squat", "Lunge"])

    st.session_state.exercise = exercise
    print(f"You selected: {st.session_state.exercise}")

    # 세션 상태 초기화
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "page" not in st.session_state:
        st.session_state.page = 1

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.header("📸 카메라로 촬영하기")
        st.write(" ")
        st.write(" ")
        # 버튼 생성 및 상태 전환
        if st.button(
            "촬영 시작" if not st.session_state.camera_active else "촬영 완료",
            on_click=toggle_camera,
        ):
            # 상태 변경에 따라 Streamlit이 자동으로 UI를 갱신
            pass

        # 카메라 활성화 상태 처리
        if st.session_state.camera_active:
            print("Camera is active. Click 'End Camera' to stop recording.")

            # 카메라 설정
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Failed to open the camera.")
                return

            st_frame = st.empty()  # 빈 프레임을 Streamlit에 생성
            video_writer = None

            # 프레임 처리
            try:
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture video.")
                        break

                    flipped_frame = cv2.flip(frame, 1)

                    # 모델로 프레임 처리
                    processed_frame = process_frame_with_model(flipped_frame)

                    # 프레임을 RGB로 변환 후 Streamlit에 표시
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    st_frame.image(frame_pil, caption="Real-time Video", use_container_width=True)

                    # 비디오 저장 설정
                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 코덱 설정
                        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        st.session_state.video_path = temp_video_file.name
                        video_writer = cv2.VideoWriter(temp_video_file.name, fourcc, 20.0, (flipped_frame.shape[1], flipped_frame.shape[0]))

                        if not video_writer.isOpened():
                            print("Error: VideoWriter failed to initialize.")
                        else:
                            print(f"VideoWriter initialized successfully. Saving to: {st.session_state.video_path}")

                    # 원본 프레임 저장
                    video_writer.write(flipped_frame)

            finally:
                cap.release()
                if video_writer:
                    video_writer.release()

    with col2:
        # 추가된 부분: 비디오 업로드 섹션
        # 비디오 업로드 옵션을 제공
        st.header("📤 운동 영상 업로드")

        uploaded_file = st.file_uploader("Upload your video file:", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.read())
                st.session_state.video_path = temp_file.name

            st.video(st.session_state.video_path)

        # 추가된 부분: 진단하기 버튼
        st.write("---")

        if st.button("진단하기", on_click=click_diagnosis):
            if not ("video_path" in st.session_state and st.session_state.video_path):
                st.warning("Please upload or record a video first.")
            # 상태 변경에 따라 Streamlit이 자동으로 UI를 갱신
            pass