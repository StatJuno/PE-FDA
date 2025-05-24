# utils/report_utils.py

from utils.test import *
from utils.model.model import inference    # inference(alias of infer_new_data)
from openai import OpenAI
from typing import List

from dotenv import load_dotenv
import os
import tempfile
import pandas as pd

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
api_key = os.getenv("GPT_API_KEY")

# GPT API 설정
client = OpenAI(api_key=api_key)
exercise_model_map = {
    "Side-Lateral-Raise": "utils/model/lateralraise_fin.pkl",
    "Lunge": "utils/model/lunge_fin.pkl"
}

def run_posture_model(video_path, exercise):
    # 1) 비디오 분석 → DataFrame 리스트 반환
    smoothed_data = process_video_and_smooth(video_path, exercise)
    segments      = segment_reps(smoothed_data)
    input_data    = combine_segments(segments)   # List[pd.DataFrame]

    # 2) 피크 프레임 이미지 생성
    first_df = input_data[0]
    min_row  = first_df.loc[first_df['LEFT_ELBOW_y'].idxmin()]
    peak_frame_number = min_row['frame_no']
    user_image   = extract_frame_as_image(video_path, peak_frame_number)
    output_image = process_pose_comparison(f"utils/gt_images/{exercise}.jpg", user_image)

    # 3) FTS 모델 인퍼런스
    model_path = exercise_model_map[exercise]
    # DataFrame 리스트 → 임시 CSV로 저장 → inference 호출
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, df_seg in enumerate(input_data):
            tmp_path = os.path.join(tmpdir, f"segment_{idx}.csv")
            df_seg.to_csv(tmp_path, index=False)
        results = inference(model_path, tmpdir, fixed_rows=300)
    # results: List[(filename, label)]
    predicted_labels = [label for _, label in results]

    # 4) GPT 리포트 생성
    report = make_report(predicted_labels, exercise)

    return report, output_image


def make_report(predictions: List[str], exercise) -> str:
    flattened = {code for code in predictions}
    prompt = exercise_prompts[exercise].format(predictions=flattened)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 유능한 피트니스 코치입니다."},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content



exercise_prompts = {
    "Side-Lateral-Raise": """
    당신은 전문 피트니스 코치입니다. 숫자 코드 형태로 1개의 reps마다 제공되는 사용자 운동 피드백을 참고하여, 전체 세트에 대한 통합적인 피드백을 작성하세요.

    다음은 숫자 코드와 사이드 레터럴 레이즈 자세 상태의 매핑 정보입니다:
    - 377: 모든 조건(무릎 반동 없음, 어깨 으쓱 없음, 상완과 전완 각도 고정, 손목 각도 고정, 상체 반동 없음)을 만족.
    - 378: 무릎 반동 있음, 나머지 조건은 만족.
    - 379: 어깨 으쓱 있음, 나머지 조건은 만족.
    - 380: 상완과 전완 각도가 고정되지 않음, 나머지 조건은 만족.
    - 381: 손목 각도가 고정되지 않음, 나머지 조건은 만족.
    - 382: 상체 반동 있음, 나머지 조건은 만족.
    
    숫자 코드:
    {predictions}

    중점을 두어야 할 부분:
    1. 운동 전반의 장점 식별.
    2. 전체 세트에서 공통으로 발견되는 문제점 또는 개선이 필요한 부분.
    3. 전체적인 자세 유지 및 부상 방지를 위한 팁.

    아래 형식에 맞춰서 출력해줘: 
    당신의 사이드 레터럴 레이즈 점수는?

    **종합 점수**: XX점 / 100점

    ### 항목별 점수
    1. **무릎 반동**: XX점 / 20점  
    - **피드백**: 

    2. **어깨 으쓱**: XX점 / 20점  
    - **피드백**:

    3. **상완과 전완 각도 고정**: XX점 / 20점  
    - **피드백**: 

    4. **손목 각도 고정**: XX점 / 20점  
    - **피드백**:

    5. **상체 안정성**: XX점 / 20점  
    - **피드백**: 

    ---

    ### **총평**
    
    
    **추천 개선 방안**:
    """,
    
    "Lunge": """
    당신은 전문 피트니스 코치입니다. 숫자 코드 형태로 1개의 reps마다 제공되는 사용자 운동 피드백을 참고하여, 전체 세트에 대한 통합적인 피드백을 작성하세요.

    다음은 숫자 코드와 런지 상태의 매핑 정보입니다:
    - 81: 모든 조건(앞다리 무릎 각도 90도, 몸통 방향 및 무릎 정렬, 뒷다리 무릎 각도 90도, 척추의 중립, 상체의 과도한 숙임/젖힘 없음)을 만족.
    - 82: 앞다리 무릎 각도 90도 아님, 나머지 조건은 만족.
    - 83: 몸통 방향 및 무릎 정렬 안 됨, 나머지 조건은 만족.
    - 84: 뒷다리 무릎 각도 90도 아님, 나머지 조건은 만족.
    - 85: 척추의 중립 없음, 나머지 조건은 만족.
    - 86: 상체의 과도한 숙임/젖힘 있음, 나머지 조건은 만족.
    
    숫자 코드:
    {predictions}

    중점을 두어야 할 부분:
    1. 운동 전반의 장점 식별.
    2. 전체 세트에서 공통으로 발견되는 문제점 또는 개선이 필요한 부분.
    3. 전체적인 자세 유지 및 부상 방지를 위한 팁.

    아래 형식에 맞춰서 출력해줘: 
    당신의 런지 점수는?

    **종합 점수**: XX점 / 100점

    ### 항목별 점수
    1. **앞다리 무릎 각도 90도**: XX점 / 20점  
    - **피드백**: 

    2. **몸통 방향 및 무릎 정렬**: XX점 / 20점  
    - **피드백**:

    3. **뒷다리 무릎 각도 90도**: XX점 / 20점  
    - **피드백**: 

    4. **척추의 중립**: XX점 / 20점  
    - **피드백**:

    5. **상체의 과도한 숙임/젖힘 여부**: XX점 / 20점
    - **피드백**: 

    ---

    ### **총평**
    
    
    **추천 개선 방안**:
    """
}
