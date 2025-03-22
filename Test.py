import random
import json
from transformers import pipeline
from huggingface_hub import login

# Hugging Face 계정에서 생성한 Access Token을 사용하여 로그인
login(token="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")

# 1. AI 캐릭터 데이터 생성 함수
def generate_ai_character_data():
    instincts = {
        "species": random.choice(["인간", "엘프", "드워프", "오크"]),
        "basic_instincts": {
            "survival_instinct": random.uniform(0.5, 1.0),
            "social_instinct": random.uniform(0.5, 1.0),
            "fear_factor": random.uniform(0.0, 1.0)
        }
    }

    # 후천적 학습 (지식과 기술 습득)
    knowledge = [
        {"source": "책 - 의약초 백과", "confidence": random.uniform(0.7, 1.0), "verified": random.choice([True, False]), "topic": random.choice(["치유 허브", "희귀 초목", "독초"])},
        {"source": "마을 장로 대화", "confidence": random.uniform(0.3, 0.7), "verified": random.choice([True, False]), "topic": random.choice(["고대 유적 위치", "전설의 보물", "영웅의 이야기"])}
    ]

    # 감정 상태 (AI의 반응을 설정)
    emotions = [
        {"emotion": "happiness", "impact_on_behavior": "social_cooperation", "duration": "3 hours"},
        {"emotion": "anger", "impact_on_behavior": "increased_aggression", "duration": "1 hour"},
        {"emotion": "sadness", "impact_on_behavior": "avoid_social_interaction", "duration": "2 hours"}
    ]

    # 목표 및 성장 시스템
    goals = {
        "current_goal": random.choice(["resource_collection", "social_interaction", "building"]),
        "goal_priority": random.uniform(0.5, 1.0),
        "long_term_goal": random.choice(["social_ranking_improvement", "village_growth", "leadership"]),
        "goal_progress": random.uniform(0.0, 1.0)
    }

    # 대화 및 기억 시스템
    recent_conversations = [
        {"partner": "NPC_1", "topic": random.choice(["거래 제안", "정치적 협상", "자원 공유"]), "outcome": random.choice(["성공", "실패"]), "emotion_impact": random.choice(["positive", "negative"])}
    ]

    long_term_memory = [
        {"topic": "협력 관계", "result": random.choice(["우호적 관계 확립", "의심스러운 관계 유지"])}
    ]

    return {
        "instincts": instincts,
        "knowledge": knowledge,
        "emotions": emotions,
        "goals": goals,
        "recent_conversations": recent_conversations,
        "long_term_memory": long_term_memory
    }

# 캐릭터 데이터 생성
ai_characters = [generate_ai_character_data() for _ in range(5)]

# 2. 모델 로딩
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline("text-generation", model=model_id, device=0)  # GPU 사용시 device=0

# 3. 상황 설정 (상황에 따라 AI 행동을 정의)
situations = [
    {"scenario": "다른 NPC와 거래 대화", "topic": "거래 제안", "emotion": "happiness", "impact_on_behavior": "social_cooperation"},
    {"scenario": "음식 채집", "topic": "배고픔", "emotion": "neutral", "impact_on_behavior": "resource_collection"},
    {"scenario": "건물 건축", "topic": "주거지 건설", "emotion": "neutral", "impact_on_behavior": "building"},
]

# 랜덤 상황 선택
random_situation = random.choice(situations)

# 4. 캐릭터 상태와 상황에 맞는 입력 데이터 생성
selected_character = ai_characters[0]  # 첫 번째 캐릭터를 선택

# 캐릭터의 상태를 포함한 입력 생성
ai_test_input = [
    {"role": "system", "content": f"당신은 {selected_character['instincts']['species']} 종족의 자율적인 AI 캐릭터입니다. 현재 감정 상태는 {random_situation['emotion']}입니다. 최근 대화에서 {random_situation['topic']}이/가 있었습니다. 목표는 {selected_character['goals']['current_goal']}이고, 목표 진행 상황은 {selected_character['goals']['goal_progress']:.2f}입니다. 현재 상태에 맞는 행동을 선택하세요."},
    {"role": "user", "content": f"현재 상황은 '{random_situation['scenario']}'입니다. 이를 바탕으로 적절한 행동을 선택하고 설명하세요."}
]

# 5. 모델을 통한 대화 내용 생성
output = pipe(ai_test_input, max_new_tokens=150)

# 6. 결과 출력
generated_text = output[0]["generated_text"]
print(f"Generated response for '{random_situation['scenario']}' with character state:")
print(generated_text)

# 7. 캐릭터 상태 변화 (예: 목표 달성도 변화)
new_goal_progress = selected_character['goals']['goal_progress'] + random.uniform(0.05, 0.1)
print(f"\nCharacter's new goal progress after '{random_situation['scenario']}': {new_goal_progress:.2f}")
