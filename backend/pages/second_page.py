import cv2
import streamlit as st
import asyncio
from PIL import Image
from utils.report_utils import run_posture_model
from utils.chat_utils import async_stream_chat_with_feedback
import numpy as np


def second_page():
    st.markdown(
        """
        <h2 style="text-align: center; font-size: 40px; margin-top: 20px;">
            운동 결과 분석 📊
        </h2>
        """,
        unsafe_allow_html=True,
    )
    st.write("---")

    # 로딩 메시지 표시
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown(
            """
            <div style="text-align: center; font-size: 20px;">
                운동을 분석 중입니다... 잠시만 기다려 주세요 ⏳
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 모델 추론
    feedback_report, feedback_image = run_posture_model(
        st.session_state.video_path,
        st.session_state.exercise
    )

    # 로딩 메시지 제거
    loading_placeholder.empty()

    # 채팅용 세션 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "new_message" not in st.session_state:
        st.session_state.new_message = None

    # 레이아웃 분할
    col1, col2 = st.columns(2)

    # 왼쪽: 분석 리포트와 이미지
    with col1:
        st.subheader("분석 리포트📄")
        st.write(feedback_report)

        st.subheader("상세 프레임 이미지")
        # feedback_image 유효성 검사
        if feedback_image is None or (isinstance(feedback_image, np.ndarray) and feedback_image.size == 0):
            st.warning("⚠️ 비교 이미지 생성에 실패했습니다. 자세 랜드마크를 감지할 수 없었어요.")
        else:
            # OpenCV 이미지를 RGB로 변환
            image_rgb = cv2.cvtColor(feedback_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            st.image(pil_image, use_container_width=True)

    # 오른쪽: 챗봇 인터페이스
    with col2:
        st.subheader("💬 챗봇과 대화하기")
        chat_placeholder = st.empty()
        with chat_placeholder.container():
            for msg in st.session_state.chat_history:
                with st.chat_message("user" if msg.get("is_user") else "assistant"):
                    st.markdown(msg.get("text", ""))
        last_placeholder = st.empty()
        user_input = st.chat_input("운동 자세에 대해 챗봇과 대화하기:")
        if user_input:
            st.session_state.new_message = user_input
            st.session_state.chat_history.append({"is_user": True, "text": user_input})
            st.session_state.chat_history.append({"is_user": False, "text": ""})
            with last_placeholder.container():
                with st.chat_message("assistant"):
                    st.markdown("")

            async def stream_response():
                resp_stream = async_stream_chat_with_feedback(
                    feedback_report,
                    st.session_state.chat_history,
                    st.session_state.new_message
                )
                assistant_text = ""
                async for chunk in resp_stream:
                    assistant_text += chunk
                    st.session_state.chat_history[-1]["text"] = assistant_text
                    with last_placeholder.container():
                        with st.chat_message("assistant"):
                            st.markdown(assistant_text)
                st.session_state.new_message = None

            asyncio.run(stream_response())

    # 뒤로가기 버튼
    def go_back():
        st.session_state.clear()
        st.session_state.page = 1
    st.button("뒤로가기", on_click=go_back)
