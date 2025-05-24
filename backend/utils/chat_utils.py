import os

import httpx
import json

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
api_key = os.getenv("GPT_API_KEY")

async def async_stream_chat_with_feedback(report, chat_history, user_input):
    """
    OpenAI GPT 스트리밍 응답 처리 함수 (최신 API 호환)
    """
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    chat_history_str = ""
    for message in chat_history:
        if message["is_user"]:
            chat_history_str += f"User: {message['text']}\n"
        else:
            chat_history_str += f"Assistant: {message['text']}\n"

    prompt = (
        f"당신은 운동 전문가입니다. 다음은 사용자가 한 운동에 관한 정리입니다: "
        f"'{report}'. 이 정보를 참고하여 운동에 대한 지식을 바탕으로 사용자의 질문에 답해주세요. "
        f" 다음은 이전 대화 내용입니다:'\n{chat_history_str}\n'"
        f"사용자의 질문은 다음과 같습니다: '{user_input}'. "
        f"질문에 대해 친절하고 자세하게 설명해주세요. 사용자가 이해할 수 있도록 구체적이고 명확한 답변을 해주세요."
    )

    # chat_history를 OpenAI API 형식에 맞게 변환
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_input}]

    payload = {
        "model": "gpt-4",
        "messages": messages,
        "stream": True  # 스트리밍 활성화
    }

    # 비동기 클라이언트로 OpenAI API 스트리밍 요청
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            async for line in response.aiter_lines():
                content = extract_content_from_stream(line)
                if content == "[DONE]":  # 스트리밍 종료 신호
                    break
                if content:  # 유효한 content만 반환
                    yield content


def extract_content_from_stream(line):
    """
    스트리밍 응답의 JSON 데이터를 파싱하고, 'content' 필드 추출

    Args:
        line (str): 스트리밍 데이터 한 줄 (e.g., 'data: {json}')

    Returns:
        str or None: 추출된 'content' 값, 없으면 None
    """
    if line.strip() and line.startswith("data:"):
        # `data:` 접두어 제거
        data = line[len("data:"):].strip()
        if data == "[DONE]":  # 스트리밍 종료 신호
            return "[DONE]"
        try:
            chunk = json.loads(data)
            if "choices" in chunk:
                delta = chunk["choices"][0].get("delta", {})
                return delta.get("content")  # 'content' 값 반환
        except json.JSONDecodeError:
            return None
    return None
