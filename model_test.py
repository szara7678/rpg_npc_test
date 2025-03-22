import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from model_config import get_model_config, ModelType, get_optimization_settings

# Hugging Face 토큰으로 로그인
login(token="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")

def load_model(model_type=ModelType.LLAMA3, model_size="3B", optimization_preset="cpu_only"):
    """
    모델 로드 및 파이프라인 설정
    
    Args:
        model_type (ModelType): 모델 타입 (LLAMA3, MISTRAL, GPT_NEO 등)
        model_size (str): 모델 크기 ("3B", "7B", "1.3B" 등)
        optimization_preset (str): 최적화 설정 ("cpu_only", "mid_range_gpu" 등)
        
    Returns:
        tuple: (model, tokenizer, pipeline)
    """
    print(f"모델 설정 로드 중: {model_type.value} {model_size}")
    model_config = get_model_config(model_type, model_size)
    optimization = get_optimization_settings(optimization_preset)
    
    model_id = model_config["model_id"]
    
    print(f"모델 로드 중: {model_id}")
    print(f"최적화 설정: {optimization_preset}")
    
    # GPU 사용 설정
    device_map = "auto" if optimization["device"] == "cuda" else "cpu"
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype="auto"
    )
    
    # 파이프라인 생성
    gen_pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer
    )
    
    return model, tokenizer, gen_pipe

def test_model_response(pipe, prompt, max_tokens=150):
    """
    모델 응답 테스트
    
    Args:
        pipe: 생성 파이프라인
        prompt (str): 프롬프트
        max_tokens (int): 최대 토큰 수
        
    Returns:
        str: 생성된 텍스트
    """
    print(f"\n프롬프트: {prompt}")
    
    start_time = time.time()
    response = pipe(prompt, max_new_tokens=max_tokens)
    end_time = time.time()
    
    generated_text = response[0]["generated_text"]
    
    print(f"\n응답 (생성 시간: {end_time - start_time:.2f}초):")
    print(generated_text)
    print("-" * 50)
    
    return generated_text

def run_model_test():
    """
    다양한 모델 테스트 실행
    """
    print("=== AI 모델 테스트 ===")
    
    # 사용 가능한 모델 타입 및 크기 표시
    print("\n사용 가능한 모델 목록:")
    for model_type in ModelType:
        print(f"- {model_type.value}")
    
    # 모델 선택
    print("\n테스트할 모델을 선택하세요:")
    print("1. Llama 3 (3B)")
    print("2. Llama 3 (8B) - 더 큰 모델, 높은 메모리 필요")
    print("3. Mistral (7B)")
    print("4. GPT-Neo (1.3B)")
    
    choice = input("선택 (기본: 1): ").strip()
    
    if choice == "2":
        model_type = ModelType.LLAMA3
        model_size = "8B"
    elif choice == "3":
        model_type = ModelType.MISTRAL
        model_size = "7B"
    elif choice == "4":
        model_type = ModelType.GPT_NEO
        model_size = "1.3B"
    else:
        # 기본값
        model_type = ModelType.LLAMA3
        model_size = "3B"
    
    # 최적화 설정 선택
    print("\n실행 환경을 선택하세요:")
    print("1. CPU Only")
    print("2. Mid-range GPU")
    print("3. High-end GPU")
    
    hw_choice = input("선택 (기본: 1): ").strip()
    
    if hw_choice == "2":
        optimization_preset = "mid_range_gpu"
    elif hw_choice == "3":
        optimization_preset = "high_end_gpu"
    else:
        # 기본값
        optimization_preset = "cpu_only"
    
    # 모델 로드
    model, tokenizer, pipe = load_model(model_type, model_size, optimization_preset)
    
    # 테스트 프롬프트 준비
    print("\n테스트할 프롬프트 타입을 선택하세요:")
    print("1. NPC 성격 및 행동 시뮬레이션")
    print("2. 환경 반응 및 의사결정")
    print("3. 대화 시뮬레이션")
    print("4. 사용자 정의 프롬프트")
    
    prompt_choice = input("선택 (기본: 1): ").strip()
    
    if prompt_choice == "2":
        # 환경 반응 프롬프트
        prompt = """<|system|>
You are an autonomous NPC in a fantasy RPG world. You make decisions based on your character traits and the current situation.

You are a Human with the following traits:
- Survival instinct: 0.8/1.0
- Social instinct: 0.7/1.0
- Fear factor: 0.4/1.0

Current state:
- Location: town_square
- Time: evening
- Weather: rainy
- Current emotion: worried
- Energy level: 70/100
- Health: 90/100

Current situation:
You've heard rumors of bandits planning to attack the town. The weather is bad and it's getting dark.

Decide what action to take based on your personality and the current situation.

You should decide on a specific action and return it in the format: ACTION: [your chosen action]
<|user|>
What will you do in this situation?
<|assistant|>"""
        
    elif prompt_choice == "3":
        # 대화 시뮬레이션 프롬프트
        prompt = """<|system|>
You are an autonomous NPC in a fantasy RPG world engaged in a conversation with another character.

You are an Elf merchant with the following traits:
- Survival instinct: 0.6/1.0
- Social instinct: 0.9/1.0
- Fear factor: 0.3/1.0

Current state:
- Location: marketplace
- Time: morning
- Current emotion: happy

You are talking to a Human adventurer about rare magical items.

Relevant memories:
You talked with this adventurer before about dragon scales. Content: The adventurer mentioned needing dragon scales for a special armor.

Respond in a conversational manner, expressing your character's thoughts, feelings, and desires based on your personality and memories.
<|user|>
The adventurer says: "Do you have any dragon scales in stock today? I need them urgently for a commission."

How do you respond?
<|assistant|>"""
        
    elif prompt_choice == "4":
        # 사용자 정의 프롬프트
        prompt = input("\n프롬프트 입력: ")
        
    else:
        # 기본: NPC 성격 및 행동 시뮬레이션
        prompt = """<|system|>
You are an autonomous NPC in a fantasy RPG world. 

You are a Dwarf blacksmith named Thorin with the following traits:
- Very skilled in metalworking
- Proud of your craft
- Somewhat grumpy but fair in dealings
- Values quality above all else

Your shop is located in the main town square. You craft weapons, armor, and tools.
Describe your typical day, your approach to your craft, and how you interact with customers.
<|user|>
Tell me about yourself and your work as a blacksmith.
<|assistant|>"""

    # 반복 테스트 옵션
    num_tests = 1
    test_option = input("\n반복 테스트 횟수 (기본: 1): ").strip()
    if test_option.isdigit():
        num_tests = int(test_option)
    
    # 테스트 실행
    for i in range(num_tests):
        if num_tests > 1:
            print(f"\n테스트 {i+1}/{num_tests}")
        test_model_response(pipe, prompt)

if __name__ == "__main__":
    run_model_test() 